import diffusers.schedulers

# scheduler dict includes superclass SchedulerMixin (it still generates reasonable images)
scheduler_dict = {
    k: v
    for k, v in diffusers.schedulers.__dict__.items()
    if "Scheduler" in k and "Flax" not in k
}
scheduler_dict.pop(
    "VQDiffusionScheduler"
)  # requires unique parameter, unlike other schedulers
scheduler_names = list(scheduler_dict.keys())
default_scheduler = scheduler_names[3]  # expected to be DPM Multistep

with open("model_ids.txt", "r") as fp:
    model_ids = fp.read().splitlines()
