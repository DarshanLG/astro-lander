Optional JSON overrides for `scripts/train.py --config <file>`. Any `PPOConfig` field can be set, e.g.

```json
{ "total_timesteps": 2000000, "ent_coef": 0.005, "env": { "continuous": true } }
```
