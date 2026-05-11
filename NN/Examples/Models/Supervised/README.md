# Supervised Examples (TorchLean)

This folder is for runnable examples where the task is ordinary input and target training:

- `Mlp.lean`: tabular supervised training on the small Auto MPG dataset.
- `LstmRegression.lean`: time series forecasting on UCI household power windows.

The architecture does not decide the folder by itself. `LstmRegression.lean` uses an LSTM, but the
example is mainly about supervised forecasting (`past window -> target window`), so it belongs here.
The `Sequence/` folder is for sequence model behavior itself: recurrent layer smoke tests,
Transformer blocks, GPT style language modeling, Mamba, and synthetic sequence curricula.

Useful commands:

```bash
python3 scripts/datasets/download_example_data.py --auto-mpg
lake exe torchlean mlp --cpu --steps 10

python3 scripts/datasets/download_example_data.py --household-power --household-power-windows 512
lake exe -K cuda=true torchlean lstm_regression --cuda --steps 200 --windows 96
```
