"""
Statistics to track agents' performance.
"""


import time
from collections import deque
import numpy as np
from tensorboardX import SummaryWriter


class Stats():
    def __init__(self):
        self.score = None
        self.avg_score = None
        self.std_dev = None
        self.scores = []                         # list containing scores from each episode
        self.avg_scores = []                     # list containing average scores after each episode
        self.scores_window = deque(maxlen=100)   # last 100 scores
        self.best_avg_score = -np.Inf            # best score for a single episode
        self.time_start = time.time()            # track cumulative wall time
        self.total_steps = 0                     # track cumulative steps taken
        self.writer = SummaryWriter()

    def update(self, steps, rewards, i_episode):
        """Update stats after each episode."""
        self.total_steps += steps
        self.score = sum(rewards)
        self.scores_window.append(self.score)
        self.scores.append(self.score)
        self.avg_score = np.mean(self.scores_window)
        self.avg_scores.append(self.avg_score)
        self.std_dev = np.std(self.scores_window)
        # update best average score
        if self.avg_score > self.best_avg_score and i_episode > 100:
            self.best_avg_score = self.avg_score

    def is_solved(self, i_episode, solve_score):
        """Define solve criteria."""
        return self.avg_score >= solve_score and i_episode >= 100

    def print_episode(self, i_episode, steps, stats_format, buffer_len, noise_weight,
                      critic_loss_01, critic_loss_02,
                      actor_loss_01, actor_loss_02,
                      noise_val_01, noise_val_02,
                      rewards_01, rewards_02):
        common_stats = 'Episode: {:5}  Avg: {:8.3f}  BestAvg: {:8.3f}  σ: {:8.3f}  |  Steps: {:8}  Reward: {:8.3f}  |  '.format(i_episode, self.avg_score, self.best_avg_score, self.std_dev, steps, self.score)
        print('\r' + common_stats + stats_format.format(buffer_len, noise_weight), end="")
        # statistics to tensorboard
        self.writer.add_scalar('global/reward', self.score, i_episode)
        self.writer.add_scalar('global/std_dev', self.std_dev, i_episode)
        self.writer.add_scalar('global/avg_reward', self.avg_score, i_episode)
        self.writer.add_scalar('global/buffer_len', buffer_len, i_episode)
        self.writer.add_scalar('global/noise_weight', noise_weight, i_episode)
        self.writer.add_scalar('agent_01/critic_loss', critic_loss_01, i_episode)
        self.writer.add_scalar('agent_02/critic_loss', critic_loss_02, i_episode)
        self.writer.add_scalar('agent_01/actor_loss', actor_loss_01, i_episode)
        self.writer.add_scalar('agent_02/actor_loss', actor_loss_02, i_episode)
        self.writer.add_scalar('agent_01/noise_val_01', noise_val_01[0], i_episode)
        self.writer.add_scalar('agent_01/noise_val_02', noise_val_01[1], i_episode)
        self.writer.add_scalar('agent_02/noise_val_01', noise_val_02[0], i_episode)
        self.writer.add_scalar('agent_02/noise_val_02', noise_val_02[1], i_episode)
        self.writer.add_scalar('agent_01/reward', rewards_01, i_episode)
        self.writer.add_scalar('agent_02/reward', rewards_02, i_episode)

    def print_epoch(self, i_episode, stats_format, *args):
        n_secs = int(time.time() - self.time_start)
        common_stats = 'Episode: {:5}   Avg: {:8.3f}   BestAvg: {:8.3f}   σ: {:8.3f}  |  Tot. Steps: {:8}   Secs: {:6}      |  '.format(i_episode, self.avg_score, self.best_avg_score, self.std_dev, self.total_steps, n_secs)
        print('\r' + common_stats + stats_format.format(*args))

    def print_solve(self, i_episode, stats_format, best, *args):
        self.print_epoch(i_episode, stats_format, *args)
        if best:
            print('\nSolved best margin in {:d} episodes!'.format(i_episode-100))
        else:
            print('\nSolved in {:d} episodes!'.format(i_episode-100))
