import csv

def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)

def save_csv(episode_lst, score_lst):
    f = open('result.csv', 'w', newline='')
    w = csv.writer(f)
    w.writerow(episode_lst)
    w.writerow(score_lst)
    f.close()
