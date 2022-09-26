import collections
import torch
import numpy as np

from ortools.sat.python import cp_model

# Named tuple to store information about created variables.
task_type = collections.namedtuple('task_type', 'start end interval')
# Named tuple to manipulate solution information.
assigned_task_type = collections.namedtuple('assigned_task_type',
                                            'start job index duration')

def eval_makespan(g, actions):
    njobs, nworkers = g.num_nodes('job'), g.num_nodes('worker')
    jdone = torch.tensor([False]*njobs)
    jendtime = torch.tensor([0]*njobs, dtype=torch.int64)
    operations = {i: [] for i in range(nworkers)}
    
    pre = -torch.ones(njobs, dtype=torch.int64)
    src, dst = g.edges(etype='precede')
    pre[dst] = src
    worker_assignment, jobs = torch.tensor(actions).T
    times = g.ndata['hv']['job'][:, 0].div(0.1, rounding_mode='trunc').to(torch.int64)

    ndone = jdone.sum()
    while not all(jdone):
        for w in range(15):
            # get next job
            jobids = jobs[worker_assignment==w]
            jobids = jobids[torch.where(~jdone[jobids])[0]]
            if len(jobids) == 0:
                continue
            j = jobids[0].item()
            if pre[j] > -1:
                if not jdone[pre[j]]:
                    continue
                dt = jendtime[pre[j]].item() - len(operations[w])
                if dt > 0:
                    operations[w]+= [-1]*dt
            operations[w]+= [j]*times[j]
            jendtime[j] = len(operations[w])
            jdone[j] = True
        if ndone == jdone.sum():
            break #gridlock
        ndone = jdone.sum()

    if any(jendtime==0):
        return -1
    return max(jendtime).item()

def g2jobdata(g, a=None):
    """
    Reformat data such that CPModel can solve for the optimal MakeSpan.
    
    g: any problem instatiation of jobShop. If initial g as returned from env.reset() is used, *a* can't be zero!
    a: list of action tuples as recieved from MDP path sampler
    
    From action log get sequence of assignments and get times from node features.
    Then deduce jobs (sequence of operations) and store in data log.
    
    Example format:
    jobs_data = [  # task = (machine_id, processing_time).
        [(0, 3), (1, 2), (2, 2)],  # Job0
        [(0, 2), (2, 1), (1, 4)],  # Job1
        [(1, 4), (2, 3)]  # Job2
    ]
    """
    
    if a is None:
        assignment = g.edges(etype='processing')[0].tolist()
        assert assignment, "Empty assignment. Pass in [actions] or full graph."
    else:
        a = torch.tensor(a)
        assignment = a[a[:, 1].argsort()][:, 0]
    times = g.ndata['hv']['job'][:, 0].div(0.1, rounding_mode='trunc').int().tolist()
    
    jobs_data = []
    prev_j = -1
    for (i, j) in torch.stack(g.edges(etype="precede")).T.tolist():
        if prev_j != i:
            jobs_data.append([(assignment[i], times[i])])
        jobs_data[-1].append((assignment[j], times[j]))
        prev_j = j

    single_jobs = [i for i in range(g.num_nodes('job')) if i not in torch.stack(g.edges(etype="precede")).unique()]
    for i in single_jobs:
        jobs_data.append([(assignment[i], times[i])])

    return jobs_data

def fjs_g2jobdata(g):
    """
    Reformat data such that CPModel can solve for the optimal MakeSpan.
    g: any problem instatiation of jobShop.
    
    Example format:
    jobs = [  # task = (processing_time, machine_id)
        [  # Job 0
            [(3, 0), (1, 1), (5, 2)],  # task 0 with 3 alternatives
            [(2, 0), (4, 1), (6, 2)],  # task 1 with 3 alternatives
            [(2, 0), (3, 1), (1, 2)],  # task 2 with 3 alternatives
        ],
        etc.
    ]
    """
    
    njobs, nworkers = g.num_nodes('job'), g.num_nodes('worker')
    def mj_list(time):
        return [(time, i) for i in range(nworkers)]
    
    times = g.ndata['hv']['job'][:, 0].div(0.1, rounding_mode='trunc').int().tolist()
    
    jobs_data = []
    prev_j = -1
    for (i, j) in torch.stack(g.edges(etype="precede")).T.tolist():
        if prev_j != i:
            jobs_data.append([mj_list(times[i])])
        jobs_data[-1].append(mj_list(times[j]))
        prev_j = j

    single_jobs = [i for i in range(g.num_nodes('job')) if i not in torch.stack(g.edges(etype="precede")).unique()]
    for i in single_jobs:
        jobs_data.append([mj_list(times[i])])
        
    return jobs_data

def fjsdata2text(jobs_data, nworkers, path):
    """
    If a *path* is given we store a text representation of the problem:
    Every row represents one job: the first number is the number of operations of that job, the second number 
    (let's say k>=1) is the number of machines that can process the first operation; then according to k,
    there are k pairs of numbers (machine,processing time) that specify which are the machines and the processing
    times; then the data for the second operation and so on...
    """
    incr_mid_by_1 = np.array([1, 0])
    lines = [' '.join([str(len(jobs_data)), str(nworkers), str(1)]) + '\n']
    for job in jobs_data:
        job_str = ' '.join([str(len(job))] +
                           [str(len(task)) + ' ' + 
                            (' '.join([str(i) for i in (np.array(task)[:, [1,0]]+incr_mid_by_1).flatten()])) for task in job])
        lines.append(job_str + '\n')
    file = open(path, "w+")
    file.writelines(lines)
    file.close()

def get_makespan(jobs_data, verbose=False):
    
    machines_count = 1 + max(task[0] for job in jobs_data for task in job)
    all_machines = range(machines_count)
    horizon = sum(task[1] for job in jobs_data for task in job)

    model = cp_model.CpModel()

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)
    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine = task[0]
            duration = task[1]
            suffix = '_%i_%i' % (job_id, task_id)
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                'interval' + suffix)
            all_tasks[job_id, task_id] = task_type(start=start_var,
                                                   end=end_var,
                                                   interval=interval_var)
            machine_to_intervals[machine].append(interval_var)

    # Create and add disjunctive constraints.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Precedences inside a job.
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.Add(all_tasks[job_id, task_id +
                                1].start >= all_tasks[job_id, task_id].end)


    # Makespan objective.
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [
        all_tasks[job_id, len(job) - 1].end
        for job_id, job in enumerate(jobs_data)
    ])
    model.Minimize(obj_var)

    # Creates the solver and solve.
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        if verbose:
            print('Solution:')
            # Create one list of assigned tasks per machine.
            assigned_jobs = collections.defaultdict(list)
            for job_id, job in enumerate(jobs_data):
                for task_id, task in enumerate(job):
                    machine = task[0]
                    assigned_jobs[machine].append(
                        assigned_task_type(start=solver.Value(
                            all_tasks[job_id, task_id].start),
                                           job=job_id,
                                           index=task_id,
                                           duration=task[1]))

            # Create per machine output lines.
            output = ''
            for machine in all_machines:
                # Sort by starting time.
                assigned_jobs[machine].sort()
                sol_line_tasks = 'Machine ' + str(machine) + ': '
                sol_line = '           '

                for assigned_task in assigned_jobs[machine]:
                    name = 'job_%i_task_%i' % (assigned_task.job,
                                               assigned_task.index)
                    # Add spaces to output to align columns.
                    sol_line_tasks += '%-15s' % name

                    start = assigned_task.start
                    duration = assigned_task.duration
                    sol_tmp = '[%i,%i]' % (start, start + duration)
                    # Add spaces to output to align columns.
                    sol_line += '%-15s' % sol_tmp

                sol_line += '\n'
                sol_line_tasks += '\n'
                output += sol_line_tasks
                output += sol_line

            # Finally print the solution found.
            print(f'Optimal Schedule Length: {solver.ObjectiveValue()}')
            print(output)
        
        return solver.ObjectiveValue(), "OPTIMAL" if status==cp_model.OPTIMAL else "FEASIBLE"
            
    if verbose:
        print('No solution found.')
        print('\nStatistics')
        print('  - conflicts: %i' % solver.NumConflicts())
        print('  - branches : %i' % solver.NumBranches())
        print('  - wall time: %f s' % solver.WallTime())
        
    return -1, "INFEASIBLE"

def get_fjs_makespan(jobs, verbose=False):
    """Solve a small flexible jobshop problem."""
    num_jobs = len(jobs)
    all_jobs = range(num_jobs)

    num_machines = max([a[1] for j in jobs for task in j for a in task])
    all_machines = range(num_machines)

    # Model the flexible jobshop problem.
    model = cp_model.CpModel()

    horizon = 0
    for job in jobs:
        for task in job:
            max_task_duration = 0
            for alternative in task:
                max_task_duration = max(max_task_duration, alternative[0])
            horizon += max_task_duration

    # print('Horizon = %i' % horizon)

    # Global storage of variables.
    intervals_per_resources = collections.defaultdict(list)
    starts = {}  # indexed by (job_id, task_id).
    presences = {}  # indexed by (job_id, task_id, alt_id).
    job_ends = []

    # Scan the jobs and create the relevant variables and intervals.
    for job_id in all_jobs:
        job = jobs[job_id]
        num_tasks = len(job)
        previous_end = None
        for task_id in range(num_tasks):
            task = job[task_id]

            min_duration = task[0][0]
            max_duration = task[0][0]

            num_alternatives = len(task)
            all_alternatives = range(num_alternatives)

            for alt_id in range(1, num_alternatives):
                alt_duration = task[alt_id][0]
                min_duration = min(min_duration, alt_duration)
                max_duration = max(max_duration, alt_duration)

            # Create main interval for the task.
            suffix_name = '_j%i_t%i' % (job_id, task_id)
            start = model.NewIntVar(0, horizon, 'start' + suffix_name)
            duration = model.NewIntVar(min_duration, max_duration,
                                       'duration' + suffix_name)
            end = model.NewIntVar(0, horizon, 'end' + suffix_name)
            interval = model.NewIntervalVar(start, duration, end,
                                            'interval' + suffix_name)

            # Store the start for the solution.
            starts[(job_id, task_id)] = start

            # Add precedence with previous task in the same job.
            if previous_end is not None:
                model.Add(start >= previous_end)
            previous_end = end

            # Create alternative intervals.
            if num_alternatives > 1:
                l_presences = []
                for alt_id in all_alternatives:
                    alt_suffix = '_j%i_t%i_a%i' % (job_id, task_id, alt_id)
                    l_presence = model.NewBoolVar('presence' + alt_suffix)
                    l_start = model.NewIntVar(0, horizon, 'start' + alt_suffix)
                    l_duration = task[alt_id][0]
                    l_end = model.NewIntVar(0, horizon, 'end' + alt_suffix)
                    l_interval = model.NewOptionalIntervalVar(
                        l_start, l_duration, l_end, l_presence,
                        'interval' + alt_suffix)
                    l_presences.append(l_presence)

                    # Link the master variables with the local ones.
                    model.Add(start == l_start).OnlyEnforceIf(l_presence)
                    model.Add(duration == l_duration).OnlyEnforceIf(l_presence)
                    model.Add(end == l_end).OnlyEnforceIf(l_presence)

                    # Add the local interval to the right machine.
                    intervals_per_resources[task[alt_id][1]].append(l_interval)

                    # Store the presences for the solution.
                    presences[(job_id, task_id, alt_id)] = l_presence

                # Select exactly one presence variable.
                model.AddExactlyOne(l_presences)
            else:
                intervals_per_resources[task[0][1]].append(interval)
                presences[(job_id, task_id, 0)] = model.NewConstant(1)

        job_ends.append(previous_end)

    # Create machines constraints.
    for machine_id in all_machines:
        intervals = intervals_per_resources[machine_id]
        if len(intervals) > 1:
            model.AddNoOverlap(intervals)

    # Makespan objective
    makespan = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(makespan, job_ends)
    model.Minimize(makespan)

    # Solve model.
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # Print final solution.
    if verbose:
        for job_id in all_jobs:
            print('Job %i:' % job_id)
            for task_id in range(len(jobs[job_id])):
                start_value = solver.Value(starts[(job_id, task_id)])
                machine = -1
                duration = -1
                selected = -1
                for alt_id in range(len(jobs[job_id][task_id])):
                    if solver.Value(presences[(job_id, task_id, alt_id)]):
                        duration = jobs[job_id][task_id][alt_id][0]
                        machine = jobs[job_id][task_id][alt_id][1]
                        selected = alt_id
                print(
                    '  task_%i_%i starts at %i (alt %i, machine %i, duration %i)' %
                    (job_id, task_id, start_value, selected, machine, duration))

        print('Solve status: %s' % solver.StatusName(status))
        print('Optimal objective value: %i' % solver.ObjectiveValue())
        print('Statistics')
        print('  - conflicts : %i' % solver.NumConflicts())
        print('  - branches  : %i' % solver.NumBranches())
        print('  - wall time : %f s' % solver.WallTime())
    
    mspan = solver.ObjectiveValue() if solver.StatusName(status) != 'INFEASIBLE' else -1
    return mspan, solver.StatusName(status)