# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import threading
import time
import timeit
import pprint
import json

import numpy as np

import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from src.core import file_writer
from src.core import prof
from src.core import vtrace

import src.models as models
import src.losses as losses

from src.env_utils import FrameStack
from src.utils import get_batch, log, create_env, create_buffers, act, create_heatmap_buffers

MinigridPolicyNet = models.MinigridPolicyNet
MinigridStateEmbeddingNet = models.MinigridStateEmbeddingNet
MinigridMLPEmbeddingNet = models.MinigridMLPEmbeddingNet
MinigridMLPTargetEmbeddingNet = models.MinigridMLPTargetEmbeddingNet

def momentum_update(model, target, ema_momentum):
    '''
    Update the key_encoder parameters through the momentum update:
    key_params = momentum * key_params + (1 - momentum) * query_params
    '''
    # For each of the parameters in each encoder
    for p_m, p_t in zip(model.parameters(), target.parameters()):
        p_m.data = p_m.data * ema_momentum + p_t.detach().data * (1. - ema_momentum)
    # For each of the buffers in each encoder
    for b_m, b_t in zip(model.buffers(), target.buffers()):
        b_m.data = b_m.data * ema_momentum + b_t.detach().data * (1. - ema_momentum)

def learn(actor_model,
          model,
          random_target_network,
          predictor_network,
          actor_encoder,
          encoder,
          batch,
          initial_agent_state, 
          initial_encoder_state,
          optimizer,
          predictor_optimizer,
          scheduler,
          flags,
          frames=None,
          lock=threading.Lock()):
    """Performs a learning (optimization) step."""
    with lock:
        count_rewards = torch.ones((flags.unroll_length, flags.batch_size), 
            dtype=torch.float32).to(device=flags.device)
        # Use the scale of square root N
        count_rewards = batch['episode_state_count'][1:].float().to(device=flags.device)

        encoded_states, unused_state = encoder(batch['partial_obs'].to(device=flags.device), initial_encoder_state, batch['done'])
        random_embedding_next, unused_state = random_target_network(encoded_states[1:].detach(), initial_agent_state)
        predicted_embedding_next, unused_state = predictor_network(encoded_states[1:].detach(), initial_agent_state)
        random_embedding, unused_state = random_target_network(encoded_states[:-1].detach(), initial_agent_state)
        predicted_embedding, unused_state = predictor_network(encoded_states[:-1].detach(), initial_agent_state)

        intrinsic_rewards_next = torch.norm(predicted_embedding_next.detach() - random_embedding_next.detach(), dim=2, p=2)
        intrinsic_rewards = torch.norm(predicted_embedding.detach() - random_embedding.detach(), dim=2, p=2)
        intrinsic_rewards = torch.clamp(intrinsic_rewards_next - flags.scale_fac * intrinsic_rewards, min=0)
        intrinsic_rewards *= (count_rewards == 1).float()

        intrinsic_reward_coef = flags.intrinsic_reward_coef
        intrinsic_rewards *= count_rewards * intrinsic_reward_coef
        
        num_samples = flags.unroll_length * flags.batch_size
        actions_flat = batch['action'][1:].reshape(num_samples).cpu().detach().numpy()
        intrinsic_rewards_flat = intrinsic_rewards.reshape(num_samples).cpu().detach().numpy()
        rnd_loss = flags.rnd_loss_coef * \
                losses.compute_rnd_loss(predicted_embedding_next, random_embedding_next.detach()) 
            
        learner_outputs, unused_state = model(batch, initial_agent_state)

        bootstrap_value = learner_outputs['baseline'][-1]

        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {
            key: tensor[:-1]
            for key, tensor in learner_outputs.items()
        }
        
        rewards = batch['reward']
            
        if flags.no_reward:
            total_rewards = intrinsic_rewards
        else:            
            total_rewards = rewards + intrinsic_rewards
        clipped_rewards = torch.clamp(total_rewards, -1, 1)
        
        discounts = (~batch['done']).float() * flags.discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch['policy_logits'],
            target_policy_logits=learner_outputs['policy_logits'],
            actions=batch['action'],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs['baseline'],
            bootstrap_value=bootstrap_value)

        pg_loss = losses.compute_policy_gradient_loss(learner_outputs['policy_logits'],
                                               batch['action'],
                                               vtrace_returns.pg_advantages)
        baseline_loss = flags.baseline_cost * losses.compute_baseline_loss(
            vtrace_returns.vs - learner_outputs['baseline'])
        entropy_loss = flags.entropy_cost * losses.compute_entropy_loss(
            learner_outputs['policy_logits'])

        total_loss = pg_loss + baseline_loss + entropy_loss + rnd_loss

        episode_returns = batch['episode_return'][batch['done']]
        stats = {
            'mean_episode_return': torch.mean(episode_returns).item(),
            'total_loss': total_loss.item(),
            'pg_loss': pg_loss.item(),
            'baseline_loss': baseline_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'rnd_loss': rnd_loss.item(),
            'mean_rewards': torch.mean(rewards).item(),
            'mean_intrinsic_rewards': torch.mean(intrinsic_rewards).item(),
            'mean_total_rewards': torch.mean(total_rewards).item(),
        }
        
        scheduler.step()
        optimizer.zero_grad()
        predictor_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
        nn.utils.clip_grad_norm_(predictor_network.parameters(), flags.max_grad_norm)
        optimizer.step()
        predictor_optimizer.step()

        actor_model.load_state_dict(model.state_dict())
        actor_encoder.load_state_dict(encoder.state_dict())
        return stats


def train(flags):  
    if flags.xpid is None:
        flags.xpid = flags.env + '-bebold-%s' % time.strftime('%Y%m%d-%H%M%S')
    plogger = file_writer.FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )

    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid,
                                         'model.tar')))

    T = flags.unroll_length
    B = flags.batch_size

    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        log.info('Using CUDA.')
        flags.device = torch.device('cuda')
    else:
        log.info('Not using CUDA.')
        flags.device = torch.device('cpu')

    env = create_env(flags)
    if flags.num_input_frames > 1:
        env = FrameStack(env, flags.num_input_frames)  

    if 'MiniGrid' in flags.env: 
        if flags.use_fullobs_policy:
            raise Exception('We have not implemented full ob policy!')
        else:
            model = MinigridPolicyNet(env.observation_space.shape, env.action_space.n)    
        random_target_network = MinigridMLPTargetEmbeddingNet().to(device=flags.device) 
        predictor_network = MinigridMLPEmbeddingNet().to(device=flags.device) 
        encoder = MinigridStateEmbeddingNet(env.observation_space.shape, flags.use_lstm)
    else:
        raise Exception('Only MiniGrid is suppported Now!')

    momentum_update(encoder.feat_extract, model.feat_extract, 0)
    momentum_update(encoder.fc, model.fc, 0)
    if flags.use_lstm:
        momentum_update(encoder.core, model.core, 0)
    
    buffers = create_buffers(env.observation_space.shape, model.num_actions, flags)
    heatmap_buffers = create_heatmap_buffers(env.observation_space.shape)
    model.share_memory()
    encoder.share_memory()
    
    initial_agent_state_buffers = []
    for _ in range(flags.num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)
    initial_encoder_state_buffers = []
    for _ in range(flags.num_buffers):
        state = encoder.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_encoder_state_buffers.append(state)

    actor_processes = []
    ctx = mp.get_context('fork')
    free_queue = ctx.Queue()
    full_queue = ctx.Queue()

    episode_state_count_dict = dict()
    train_state_count_dict = dict()
    partial_state_count_dict = dict()
    encoded_state_count_dict = dict()
    heatmap_dict = dict()
    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(i, free_queue, full_queue, model, encoder, buffers, 
                episode_state_count_dict, train_state_count_dict, partial_state_count_dict, encoded_state_count_dict,
                heatmap_dict, heatmap_buffers, initial_agent_state_buffers, initial_encoder_state_buffers, flags))
        actor.start()
        actor_processes.append(actor)

    if 'MiniGrid' in flags.env: 
        if flags.use_fullobs_policy:
            raise Exception('We have not implemented full ob policy!')
        else:
            learner_model = MinigridPolicyNet(env.observation_space.shape, env.action_space.n)\
                .to(device=flags.device)
            learner_encoder = MinigridStateEmbeddingNet(env.observation_space.shape, flags.use_lstm)\
                .to(device=flags.device)
    else:
        raise Exception('Only MiniGrid is suppported Now!')
    learner_encoder.load_state_dict(encoder.state_dict())

    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha)

    predictor_optimizer = torch.optim.Adam(
        predictor_network.parameters(), 
        lr=flags.predictor_learning_rate)

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_frames) / flags.total_frames

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger = logging.getLogger('logfile')
    stat_keys = [
        'total_loss',
        'mean_episode_return',
        'pg_loss',
        'baseline_loss',
        'entropy_loss',
        'rnd_loss',
        'mean_rewards',
        'mean_intrinsic_rewards',
        'mean_total_rewards',
    ]

    logger.info('# Step\t%s', '\t'.join(stat_keys))

    frames, stats = 0, {}

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, stats
        timings = prof.Timings()
        while frames < flags.total_frames:
            timings.reset()
            batch, agent_state, encoder_state = get_batch(free_queue, full_queue, buffers, 
                initial_agent_state_buffers, initial_encoder_state_buffers, flags, timings)
            stats = learn(model, learner_model, random_target_network, predictor_network,
                          encoder, learner_encoder, batch, agent_state, encoder_state, optimizer, 
                          predictor_optimizer, scheduler, flags, frames=frames)
            timings.time('learn')
            with lock:
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                frames += T * B

        if i == 0:
            log.info('Batch and learn: %s', timings.summary())

    for m in range(flags.num_buffers):
        free_queue.put(m)

    threads = []    
    for i in range(flags.num_threads):
        thread = threading.Thread(
            target=batch_and_learn, name='batch-and-learn-%d' % i, args=(i,))
        thread.start()
        threads.append(thread)


    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        checkpointpath = os.path.expandvars(os.path.expanduser(
            '%s/%s/%s' % (flags.savedir, flags.xpid,'model_'+str(frames)+'.tar')))
        log.info('Saving checkpoint to %s', checkpointpath)
        torch.save({
            'model_state_dict': model.state_dict(),
            'encoder': encoder.state_dict(),
            'random_target_network_state_dict': random_target_network.state_dict(),
            'predictor_network_state_dict': predictor_network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'predictor_optimizer_state_dict': predictor_optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'flags': vars(flags),
        }, checkpointpath)

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while frames < flags.total_frames:
            start_frames = frames
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > flags.save_interval * 60:  
                checkpoint(frames)
                save_heatmap(frames)
                last_checkpoint_time = timer()

            fps = (frames - start_frames) / (timer() - start_time)
            
            if stats.get('episode_returns', None):
                mean_return = 'Return per episode: %.1f. ' % stats[
                    'mean_episode_return']
            else:
                mean_return = ''

            total_loss = stats.get('total_loss', float('inf'))
            if stats:
                log.info('After %i frames: loss %f @ %.1f fps. Mean Return %.1f. \n Stats \n %s', \
                        frames, total_loss, fps, stats['mean_episode_return'], pprint.pformat(stats))

    except KeyboardInterrupt:
        return  
    else:
        for thread in threads:
            thread.join()
        log.info('Learning finished after %d frames.', frames)
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint(frames)
    plogger.close()

