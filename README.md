# Chess-FPT
A small GPT-NeoX based transformer model for chess play, trained on high-quality chess games in PGN format.

[View on Huggingface](https://huggingface.co/huyvux3005/chessllm_FPT)

# Web Version
This model can be run within a browser, thanks to Huggingface transformers.js!
You can try it [here](https://huyvux12.github.io/chess-llama)


# Performance
It uses the UCI format for input and output. It has been trained with the token indicating result appended to the beginning of the games, hoping it would improve performance during actual chess play. The model achieves an estimated Elo rating of 1400, and easily outperforms Skill-level 0 Stockfish, but loses to Stockfish set to level higher than 1.

![Analysis](public/vs.png)