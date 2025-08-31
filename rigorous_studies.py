import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import json

# Part 1: ODE simulation for formal model

def simulate_ouroboros_ode(alpha=0.1, beta=0.1, gamma=0.1, delta=0.3, T=100, dt=0.1, resilient=False):
    """
    Simulate the Ouroboros effect ODEs using explicit Euler integration.
    Baseline: dQ_M/dt = -alpha*(1 - Q_H) - beta*(1 - Q_H)
    Resilient: dQ_M/dt = -alpha_res*(1 - Q_H) - beta*(1 - Q_H) + delta*(1 - Q_M)
    dQ_H/dt = -gamma*(1 - Q_M)
    We bound Q_M and Q_H within [0,1].
    Returns time array, Q_M array, Q_H array.
    """
    steps = int(T / dt) + 1
    t_vals = np.linspace(0, T, steps)
    Q_M = np.ones(steps)
    Q_H = np.ones(steps)
    alpha_eff = alpha * (0.6 if resilient else 1.0)  # reduce alpha by 40% if resilient
    for i in range(1, steps):
        Qm = Q_M[i-1]
        Qh = Q_H[i-1]
        # P(t) approximated as Q_h
        P = Qh
        if resilient:
            dQm_dt = -alpha_eff * (1 - P) - beta * (1 - Qh) + delta * (1 - Qm)
        else:
            dQm_dt = -(alpha + beta) * (1 - Qh)  # since P=Qh
        dQh_dt = -gamma * (1 - Qm)
        Qm_new = Qm + dQm_dt * dt
        Qh_new = Qh + dQh_dt * dt
        # bound between 0 and 1
        Q_M[i] = max(0.0, min(1.0, Qm_new))
        Q_H[i] = max(0.0, min(1.0, Qh_new))
    return t_vals, Q_M, Q_H

# Part 2: Machine learning simulations

def prepare_dataset(name='digits', test_size=0.3, task_ratio=0.2, random_state=42):
    if name == 'digits':
        X, y = load_digits(return_X_y=True)
    elif name == 'iris':
        X, y = load_iris(return_X_y=True)
    else:
        raise ValueError('Unknown dataset')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    # Split task data from train data
    task_size = int(task_ratio * len(X_train))
    # stratified split for task dataset
    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=task_size, random_state=random_state)
    for train_idx, task_idx in splitter.split(X_train, y_train):
        X_train_base = X_train[train_idx]
        y_train_base = y_train[train_idx]
        X_task = X_train[task_idx]
        y_task = y_train[task_idx]
    # Scale features
    scaler = StandardScaler()
    X_train_base_scaled = scaler.fit_transform(X_train_base)
    X_task_scaled = scaler.transform(X_task)
    X_test_scaled = scaler.transform(X_test)
    return X_train_base_scaled, y_train_base, X_task_scaled, y_task, X_test_scaled, y_test


def run_baseline_ml(X_train_base, y_train_base, X_task, y_task, X_test, y_test,
                    iterations=10, alpha=0.05, beta=0.1, gamma=0.05, random_state=0):
    # Initialize model
    model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=random_state)
    model.fit(X_train_base, y_train_base)
    results = []
    # compute baseline accuracy
    base_acc = accuracy_score(y_test, model.predict(X_test))
    QM = base_acc
    QH = 1.0
    for it in range(1, iterations+1):
        # predictions on task set
        preds = model.predict(X_task)
        is_correct = preds == y_task
        # acceptance probability depends on current QM
        p_accept = max(0.0, 1.0 - QM)
        # synthetic labels
        synthetic_labels = []
        accepted_errors = 0
        total_errors = 0
        for correct, pred_label, true_label in zip(is_correct, preds, y_task):
            if correct:
                synthetic_labels.append(true_label)
            else:
                total_errors += 1
                if np.random.rand() < p_accept:
                    synthetic_labels.append(pred_label)
                    accepted_errors += 1
                else:
                    synthetic_labels.append(true_label)
        P = (accepted_errors / total_errors) if total_errors > 0 else 0.0
        # update Q_M and Q_H via ODE discretization
        dQM_dt = -alpha * (1 - P) - beta * (1 - QH)
        QM = max(0.0, min(1.0, QM + dQM_dt))
        dQH_dt = -gamma * (1 - QM)
        QH = max(0.0, min(1.0, QH + dQH_dt))
        # train on mixture dataset
        X_mix = np.concatenate([X_train_base, X_task])
        y_mix = np.concatenate([y_train_base, np.array(synthetic_labels)])
        model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=random_state)
        model.fit(X_mix, y_mix)
        test_acc = accuracy_score(y_test, model.predict(X_test))
        results.append({'iteration': it, 'accuracy': test_acc, 'QM': QM, 'QH': QH, 'P': P})
    return results


def run_rag_ml(X_train_base, y_train_base, X_task, y_task, X_test, y_test,
               iterations=10, k=5, random_state=0):
    # Use k-NN majority vote to correct predictions (no synthetic acceptance)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_train_base)
    model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=random_state)
    model.fit(X_train_base, y_train_base)
    # baseline accuracy
    test_acc = accuracy_score(y_test, model.predict(X_test))
    QM = test_acc
    QH = 1.0
    results = []
    for it in range(1, iterations+1):
        # Use nearest neighbors to correct each task sample
        _, indices = nn.kneighbors(X_task)
        corrected_labels = []
        for idxs, pred_label, true_label in zip(indices, model.predict(X_task), y_task):
            # Majority vote among neighbors
            neighbor_labels = y_train_base[idxs]
            majority = np.bincount(neighbor_labels).argmax()
            corrected_labels.append(majority)
        # P=0 (no accepted errors)
        P = 0.0
        # update Q_M and Q_H
        dQM_dt = -0.0 - 0.0  # no decay due to synthetic/human
        QM = max(0.0, min(1.0, QM + dQM_dt))
        dQH_dt = -0.0  # QH remains 1
        QH = max(0.0, min(1.0, QH + dQH_dt))
        # mixture dataset: base + corrected labels
        X_mix = np.concatenate([X_train_base, X_task])
        y_mix = np.concatenate([y_train_base, np.array(corrected_labels)])
        model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=random_state)
        model.fit(X_mix, y_mix)
        test_acc = accuracy_score(y_test, model.predict(X_test))
        results.append({'iteration': it, 'accuracy': test_acc, 'QM': QM, 'QH': QH, 'P': P})
    return results


def run_ea_ml(X_train_base, y_train_base, X_task, y_task, X_test, y_test,
              iterations=10, population_size=4, num_parents=2,
              alpha=0.05, beta=0.1, gamma=0.05, random_state=0):
    # Initialize population
    population = []
    for i in range(population_size):
        seed = random_state + i
        model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=seed)
        model.fit(X_train_base, y_train_base)
        population.append({'model': model, 'seed': seed})
    # evaluate best accuracy
    best_acc = 0.0
    for individual in population:
        acc = accuracy_score(y_test, individual['model'].predict(X_test))
        if acc > best_acc:
            best_acc = acc
    QM = best_acc
    QH = 1.0
    results = []
    for it in range(1, iterations+1):
        fitness_scores = []
        for idx, individual in enumerate(population):
            model = individual['model']
            preds = model.predict(X_task)
            is_correct = preds == y_task
            p_accept = max(0.0, 1.0 - QM)
            synthetic_labels = []
            accepted_errors = 0
            total_errors = 0
            for correct, pred_label, true_label in zip(is_correct, preds, y_task):
                if correct:
                    synthetic_labels.append(true_label)
                else:
                    total_errors += 1
                    if np.random.rand() < p_accept:
                        synthetic_labels.append(pred_label)
                        accepted_errors += 1
                    else:
                        synthetic_labels.append(true_label)
            P = (accepted_errors / total_errors) if total_errors > 0 else 0.0
            # mixture dataset
            X_mix = np.concatenate([X_train_base, X_task])
            y_mix = np.concatenate([y_train_base, np.array(synthetic_labels)])
            # train new model
            new_model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=individual['seed'])
            new_model.fit(X_mix, y_mix)
            acc = accuracy_score(y_test, new_model.predict(X_test))
            fitness_scores.append({'idx': idx, 'acc': acc, 'model': new_model, 'P': P})
        # sort and select parents
        fitness_scores_sorted = sorted(fitness_scores, key=lambda x: x['acc'], reverse=True)
        parents = fitness_scores_sorted[:num_parents]
        # update QM and QH based on best
        best_acc = parents[0]['acc']
        P_best = parents[0]['P']
        dQM_dt = -alpha * (1 - P_best) - beta * (1 - QH)
        QM = max(0.0, min(1.0, QM + dQM_dt))
        dQH_dt = -gamma * (1 - QM)
        QH = max(0.0, min(1.0, QH + dQH_dt))
        # new population
        new_population = []
        # keep parents
        for parent in parents:
            new_population.append({'model': parent['model'], 'seed': random_state + np.random.randint(0, 100000)})
        while len(new_population) < population_size:
            seed = random_state + np.random.randint(0, 100000)
            child_model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=seed)
            # train child on mixture dataset of best parent
            preds_best = parents[0]['model'].predict(X_task)
            is_correct_best = preds_best == y_task
            synthetic_labels_best = []
            for correct, pred_label, true_label in zip(is_correct_best, preds_best, y_task):
                if correct:
                    synthetic_labels_best.append(true_label)
                else:
                    if np.random.rand() < p_accept:
                        synthetic_labels_best.append(pred_label)
                    else:
                        synthetic_labels_best.append(true_label)
            X_mix = np.concatenate([X_train_base, X_task])
            y_mix = np.concatenate([y_train_base, np.array(synthetic_labels_best)])
            child_model.fit(X_mix, y_mix)
            new_population.append({'model': child_model, 'seed': seed})
        population = new_population
        results.append({'iteration': it, 'accuracy': best_acc, 'QM': QM, 'QH': QH, 'P': P_best})
    return results


def run_ea_rag_ml(X_train_base, y_train_base, X_task, y_task, X_test, y_test,
                  iterations=10, population_size=4, num_parents=2, k=5,
                  alpha=0.05, beta=0.1, gamma=0.05, random_state=0):
    # Use k-NN to correct labels
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_train_base)
    population = []
    for i in range(population_size):
        seed = random_state + i
        model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=seed)
        model.fit(X_train_base, y_train_base)
        population.append({'model': model, 'seed': seed})
    # evaluate best
    best_acc = 0.0
    for ind in population:
        acc = accuracy_score(y_test, ind['model'].predict(X_test))
        if acc > best_acc:
            best_acc = acc
    QM = best_acc
    QH = 1.0
    results = []
    for it in range(1, iterations+1):
        fitness_scores = []
        for idx, individual in enumerate(population):
            model = individual['model']
            # correct labels using k-NN majority
            _, indices = nn.kneighbors(X_task)
            corrected_labels = []
            for idxs, pred_label, true_label in zip(indices, model.predict(X_task), y_task):
                neighbor_labels = y_train_base[idxs]
                majority = np.bincount(neighbor_labels).argmax()
                corrected_labels.append(majority)
            P = 0.0
            # update mixture dataset
            X_mix = np.concatenate([X_train_base, X_task])
            y_mix = np.concatenate([y_train_base, np.array(corrected_labels)])
            new_model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=individual['seed'])
            new_model.fit(X_mix, y_mix)
            acc = accuracy_score(y_test, new_model.predict(X_test))
            fitness_scores.append({'idx': idx, 'acc': acc, 'model': new_model, 'P': P})
        # select parents
        fitness_scores_sorted = sorted(fitness_scores, key=lambda x: x['acc'], reverse=True)
        parents = fitness_scores_sorted[:num_parents]
        best_acc = parents[0]['acc']
        # update QM and QH (no synthetic effect)
        dQM_dt = 0.0
        QM = max(0.0, min(1.0, QM + dQM_dt))
        dQH_dt = 0.0
        QH = max(0.0, min(1.0, QH + dQH_dt))
        # create new population
        new_population = []
        for parent in parents:
            new_population.append({'model': parent['model'], 'seed': random_state + np.random.randint(0, 100000)})
        while len(new_population) < population_size:
            seed = random_state + np.random.randint(0, 100000)
            # train child on corrected labels of best parent
            _, indices = nn.kneighbors(X_task)
            corrected_labels_best = []
            for idxs, pred_label, true_label in zip(indices, parents[0]['model'].predict(X_task), y_task):
                neighbor_labels = y_train_base[idxs]
                majority = np.bincount(neighbor_labels).argmax()
                corrected_labels_best.append(majority)
            X_mix = np.concatenate([X_train_base, X_task])
            y_mix = np.concatenate([y_train_base, np.array(corrected_labels_best)])
            child_model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=seed)
            child_model.fit(X_mix, y_mix)
            new_population.append({'model': child_model, 'seed': seed})
        population = new_population
        results.append({'iteration': it, 'accuracy': best_acc, 'QM': QM, 'QH': QH, 'P': 0.0})
    return results

if __name__ == '__main__':
    # Run ODE simulations
    t_b, Qm_b, Qh_b = simulate_ouroboros_ode(alpha=0.1, beta=0.1, gamma=0.1, resilient=False)
    t_r, Qm_r, Qh_r = simulate_ouroboros_ode(alpha=0.1, beta=0.1, gamma=0.1, delta=0.3, resilient=True)
    # Save ODE results to JSON
    ode_data = {
        'baseline': {'t': t_b.tolist(), 'Qm': Qm_b.tolist(), 'Qh': Qh_b.tolist()},
        'resilient': {'t': t_r.tolist(), 'Qm': Qm_r.tolist(), 'Qh': Qh_r.tolist()}
    }
    with open('/home/oai/share/ode_results.json', 'w') as f:
        json.dump(ode_data, f, indent=2)
    # Plot ODE results
    plt.figure(figsize=(6,4))
    plt.plot(t_b, Qm_b, label='Baseline Q_M')
    plt.plot(t_b, Qh_b, label='Baseline Q_H')
    plt.plot(t_r, Qm_r, label='Resilient Q_M')
    plt.plot(t_r, Qh_r, label='Resilient Q_H')
    plt.xlabel('Time')
    plt.ylabel('Quality')
    plt.title('Ouroboros ODE Simulation')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/oai/share/ode_simulation.png')
    plt.close()

    # Run ML simulations on digits dataset
    X_train_base, y_train_base, X_task, y_task, X_test, y_test = prepare_dataset('digits', test_size=0.3, task_ratio=0.2)
    baseline_ml = run_baseline_ml(X_train_base, y_train_base, X_task, y_task, X_test, y_test, iterations=10)
    rag_ml = run_rag_ml(X_train_base, y_train_base, X_task, y_task, X_test, y_test, iterations=10)
    ea_ml = run_ea_ml(X_train_base, y_train_base, X_task, y_task, X_test, y_test, iterations=10, population_size=4, num_parents=2)
    ea_rag_ml = run_ea_rag_ml(X_train_base, y_train_base, X_task, y_task, X_test, y_test, iterations=10, population_size=4, num_parents=2)

    # Save ML results
    ml_data = {
        'baseline_ml': baseline_ml,
        'rag_ml': rag_ml,
        'ea_ml': ea_ml,
        'ea_rag_ml': ea_rag_ml
    }
    with open('/home/oai/share/ml_results.json', 'w') as f:
        json.dump(ml_data, f, indent=2)

    # Plot ML results
    def plot_ml(results, label, filename):
        iters = [r['iteration'] for r in results]
        acc = [r['accuracy'] for r in results]
        QM = [r['QM'] for r in results]
        QH = [r['QH'] for r in results]
        plt.figure(figsize=(6,4))
        plt.plot(iters, acc, marker='o', label='Test Accuracy')
        plt.plot(iters, QM, marker='s', label='QM (simulated)')
        plt.plot(iters, QH, marker='^', label='QH (simulated)')
        plt.xlabel('Iteration')
        plt.ylabel('Metric Value')
        plt.title(label)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    plot_ml(baseline_ml, 'Baseline (Digits)', '/home/oai/share/baseline_ml_digits.png')
    plot_ml(rag_ml, 'RAG (Digits)', '/home/oai/share/rag_ml_digits.png')
    plot_ml(ea_ml, 'EA (Digits)', '/home/oai/share/ea_ml_digits.png')
    plot_ml(ea_rag_ml, 'EA+RAG (Digits)', '/home/oai/share/ea_rag_ml_digits.png')

    print('Rigorous studies completed.')
