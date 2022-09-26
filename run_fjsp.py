import pdb

import argparse

import dgl
import matplotlib.pyplot as plt
import numpy as np
import timeit

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import copy, deepcopy
from torch.optim import Adam

from env.fjsp import jobShopScheduling
from actor.fjsp import *
from utils.fjsp import * 
from utils.replay_buffer import replayBuffer, sample_episode
from utils.path_collector import MdpPathCollector

def parse_args():
    parser = argparse.ArgumentParser(description="Flexible job shop problem default settings.")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--n_workers", type=int, default=15, help="number of workers available for processing")
    parser.add_argument("--n_jobs", type=int, default=25, help="number of operations to process")
    
    parser.add_argument("--eps", type=float, default=0.1, help="epislon greedy exploration parameter")
    parser.add_argument("--replay_buffer_cap", type=int, default=5000, help="replay buffer size, max nr of past samples")
    parser.add_argument("--n_samples", type=int, default=128, help="number of trajectories sampled at each epoch.")
    parser.add_argument("--prioritized_replay", type=bool, default=True, help="use prioritzed replay when sampling for HER")
    parser.add_argument("-lr", "--learning_rate", type=float, default=8e-5, help="learning rate")
    parser.add_argument("-ep", "--n_epoch", type=int, default=300, help="number of epochs")
    parser.add_argument("-it", "--n_iter", type=int, default=64, help="number of iterations per epoch")
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="batch size per iteration")
    parser.add_argument("-γ", "--gamma", type=float, default=1.0, help="discount factor")
    parser.add_argument("--layer_size", type=int, default=16, help="number of heads per GATv2 layer.")
    parser.add_argument("--k", type=int, default=2, help="number of loops over GNN layer.")
    
    parser.add_argument("--load_from", type=str, default="", help="load a pretrained network's weights")
    parser.add_argument("--save", type=bool, default=True, help="save last model instance")
    parser.add_argument("--plot", type=bool, default=True, help="plot training performance curves")
    parser.add_argument("--just_eval", type=bool, default=False, help="laod and evaluate the given model instance. No training.")

    args=parser.parse_args()
    return args


# #### Helpers

scientific_notation =  lambda x:"{:.2e}".format(x)

def get_scores(g, scores):
    n = scores.shape[0]
    idx = (g.ndata['hv']['job'][:, 3] == 0).view(n, -1)
    
    values, _ = scores.max(-1, keepdims=False)
    return torch.stack([values[i][idx[i]].max() if sum(idx[i]).item()>0 else torch.tensor(0.) for i in range(n)])

def mean_reward(paths):
    return torch.tensor([p['rewards'] for p in paths]).sum(1).mean().item()

def mean_makespan(paths):
    "Returns the average makespan successful paths from given list. Returns *nan* if no path was successful."
    return torch.tensor([p['makespan'] for p in paths if p['success']]).to(torch.float).mean().item()

def relative_makespan_error(paths, fjs=False, _dir=""):
    """ From initial conditions of each path, evaluate optimal makespan for each path and compare against
    sampled trajectory. """
    err = []
    for i, p in enumerate(paths):
        if not p['success']:
            continue
        g0 = p['observations'][0]
        jdata = fjs_g2jobdata(g0) if fjs else g2jobdata(g0, p['actions'])
        makespan, status = get_fjs_makespan(jdata) if fjs else get_makespan(jdata)
        if _dir:
            pm = int(p['makespan'])
            fjsdata2text(jdata, 
                         g0.num_nodes('worker'),
                         _dir+f"sample_{i+1}_m_opt={int(makespan)}_m_gnn={pm}.txt")
        if makespan > -1: # feasible
            relative_error = p['makespan']/makespan - 1
            err.append(relative_error)
        else: 
            print("This should not be possible, check for bugs!!")
        
    if _dir:
        print("saved", len(paths), "fsjp samples to", _dir)
    return torch.tensor(err).mean().item()


def performance_eval(env, qf, n_samples, max_len, verbose=True):
    qf.eval()
    
    eval_policy = epsilonGreedyPolicy(qf, 0.)
    eval_path_collector = MdpPathCollector(env, eval_policy, rollout_fn=sample_episode, parallelize=False)
    
    start = timeit.default_timer()
    paths = eval_path_collector.collect_new_paths(n_samples, max_len, False)
    avg_dt = (timeit.default_timer() - start) / n_samples
    
    # 1) reward
    eval_r = mean_reward(paths)
    # 2) successrate
    success_rate = np.mean([p['success'] for p in paths])
    # 3) makespan
    avg_makespan = mean_makespan(paths)
    
    start = timeit.default_timer()
    relative_err = relative_makespan_error(paths, True, _dir="data/fsjp_metaheuristic/")
    avg_or_dt = (timeit.default_timer() - start) / n_samples
    
    if verbose:
        err = 3
        print("Avg. rewards:", round(eval_r, err),
              "| Success rate:", round(success_rate, err), 
              "| Makespan:", round(avg_makespan, err), 
              "| Rel. error:", round(relative_err, err), "\n",
              "| Avg. runtime (GNN):", round(avg_dt, err), 
              "| Avg. runtime (OR):", round(avg_or_dt, err)
             )
        
        
    return eval_r, success_rate, relative_err

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    njobs, nworkers = args.n_jobs, args.n_workers
    env = jobShopScheduling(njobs, nworkers)
    env.reset()

    #load_from = args.load_from #"jobshop_qf_j25-w15_4x4_multilayer-from_script"
    if args.just_eval:
        assert args.load_from, "No model given to evaluate!"
        print("Evaluating", args.load_from, f"on {args.n_samples} sample instances.")
        qf = hgnn(args.layer_size)
        qf.load_state_dict(torch.load("chkpt/job_shop/"+args.load_from))
        performance_eval(env, qf, args.n_samples, njobs+1)
        return
        
    assert not args.just_eval, "Code should not run past this point."

    qf = hgnn(args.layer_size, k=args.k)
    target_qf = hgnn(args.layer_size, k=args.k)
    if args.load_from:
        qf.load_state_dict(torch.load("chkpt/job_shop/"+args.load_from))
        target_qf.load_state_dict(torch.load("chkpt/job_shop/"+args.load_from))
        
    expl_policy = epsilonGreedyPolicy(qf, args.eps)
    eval_policy = epsilonGreedyPolicy(target_qf, 0.)

    expl_path_collector = MdpPathCollector(env, expl_policy, rollout_fn=sample_episode, parallelize=False)
    eval_path_collector = MdpPathCollector(env, eval_policy, rollout_fn=sample_episode, parallelize=False)

    replay_buffer = replayBuffer(args.replay_buffer_cap, prioritized=args.prioritized_replay)

    optimizer = Adam(qf.parameters(), lr=args.learning_rate, weight_decay=0.01)
    qf_criterion = nn.MSELoss()

    max_len = njobs+1
    n_epoch = args.n_epoch
    n_iter = args.n_iter

    loss = []
    avg_r_train = []
    avg_r_eval = []
    success_rates = []
    relative_errors = []

    for i in range(n_epoch):
        qf.train(False)
        paths = expl_path_collector.collect_new_paths(args.n_samples, max_len, False)
        train_r = mean_reward(paths)
        avg_r_train.append(train_r)
        replay_buffer.add_paths(paths)

        paths = eval_path_collector.collect_new_paths(args.n_samples//4, max_len, False)
        eval_r = mean_reward(paths)
        avg_r_eval.append(eval_r)

        success_rate = np.mean([p['success'] for p in paths])
        success_rates.append(success_rate)

        avg_makespan = mean_makespan(paths)
        relative_err = relative_makespan_error(paths, True)
        relative_errors.append(relative_err)

        qf.train(True)
        for _ in range(n_iter):
            batch = replay_buffer.random_batch(args.batch_size)

            rewards = torch.tensor([b.r for b in batch])
            terminals = torch.tensor([b.d for b in batch]).float()
            actions = torch.tensor([b.a for b in batch])

            states = batch_graphs([b.s for b in batch])
            next_s = batch_graphs([b.sp for b in batch])        

            out = target_qf(next_s) # shape = (|G|, |J|, |W|)
            target_q_values = get_scores(next_s, out)
            y_target = rewards + (1. - terminals) * args.gamma * target_q_values 

            out = qf(states)
            y_pred = out[torch.arange(args.batch_size), actions.T[1], actions.T[0]]
            qf_loss = qf_criterion(y_pred, y_target).to(torch.float)

            loss.append(qf_loss.item())

            optimizer.zero_grad()
            qf_loss.backward()
            optimizer.step()

        target_qf.load_state_dict(deepcopy(qf.state_dict()))
        err = 3
        print("Epoch", i+1,
              " -> Loss:", round(np.mean(loss[-n_iter:]), err),
              "| Rewards: (train)", round(train_r, err), "(test)", round(eval_r, err),
              "| Success rate:", round(success_rate, err), 
              "| Makespan:", round(avg_makespan, err), 
              "| Rel. error:", round(relative_err, err), )


    target_qf.eval()
    if args.save:
        torch.save(target_qf.state_dict(), "chkpt/fjsp/jobshop_qf_j%d-w%d" % (njobs, nworkers))
    
    performance_eval(env, target_qf, args.n_samples//4, max_len)

    if args.plot:
        losses = [np.mean(loss[i*n_iter:(i+1)*n_iter]) for i in range(n_epoch)]
        x = np.arange(n_epoch)

        plt.figure(figsize=(15, 10))

        plt.subplot(221)
        plt.plot(x, losses)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.subplot(222)
        plt.plot(x, avg_r_train, label="test")
        plt.plot(x, avg_r_eval, 'r--', label="eval")
        plt.legend()
        plt.ylabel('Train/Test Rewards [path, avg]')
        plt.xlabel('Epoch')
        plt.subplot(223)
        plt.plot(x, success_rates)
        plt.ylabel('Success Rate')
        plt.xlabel('Epoch')
        plt.subplot(224)
        plt.plot(x, [0.0]*len(x), 'lightgray', linestyle='--')
        plt.plot(x, relative_errors)
        plt.ylabel('Relative Error')
        plt.xlabel('Epoch')
        plt.suptitle('Training Performance Summary', y=.95)
        plt.savefig('figs/fjsp/j%d-w%d_4x4' % (njobs, nworkers), dpi=300)

if __name__ == '__main__':
    args=parse_args()
    main(args)