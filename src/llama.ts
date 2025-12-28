import { GPTNeoXForCausalLM, AutoTokenizer, topk } from '@huggingface/transformers';
import { Chess } from 'chess.js';

let tokenizer = await AutoTokenizer.from_pretrained('huyvux3005/chessllm_FPT');
let model = await GPTNeoXForCausalLM.from_pretrained('huyvux3005/chessllm_FPT', {device: "wasm", dtype: "fp32"});

let k = tokenizer.model.vocab.length

let mid = 0;

const diffSelector:HTMLSelectElement = document.getElementById("difficulty")! as HTMLSelectElement;
diffSelector.addEventListener("change", () => {
	mid = parseInt(diffSelector.value);
})

async function wait(ms: number): Promise<void> {
	return new Promise(resolve => setTimeout(resolve, ms));
}

// Convert SAN moves array to PGN format with move numbers
// e.g., ["e4", "e5", "Nf3", "Nc6"] -> "1. e4 e5 2. Nf3 Nc6"
function toPGN(sanMoves: string[]): string {
	let pgn = "";
	for (let i = 0; i < sanMoves.length; i++) {
		if (i % 2 === 0) {
			// White's move - add move number
			pgn += `${Math.floor(i / 2) + 1}. ${sanMoves[i]} `;
		} else {
			// Black's move
			pgn += `${sanMoves[i]} `;
		}
	}
	return pgn.trim();
}

// sanMoves: array of SAN moves (e.g., ["e4", "e5", "Nf3"])
// chess: current Chess instance to validate moves
export default async function playLlama(sanMoves: string[], chess: Chess): Promise<string> {
	const start = performance.now();
	
	// Convert to PGN format with move numbers
	const inputText = toPGN(sanMoves);
	console.log("Input to model:", inputText);
	
	let inputs = await tokenizer(inputText);
	let { logits } = await model(inputs);
	let preds = logits.slice(null, -1, null);

	const [_v, {data}] = await topk(preds, k);
	const end = performance.now();
	console.log(`Inference took ${end - start} ms`);

	if(end - start < 1000) {
		await wait(1000 - (end - start));
	}

	// Try to find a valid SAN move from model predictions
	for(let i = mid; i >= 0; i--) {
		const sanMove = tokenizer.decode([data[i]]).trim();
		try {
			// Validate move by trying it on a copy of the chess instance
			const testChess = new Chess(chess.fen());
			const move = testChess.move(sanMove);
			if (move) {
				console.log(`Found valid move: ${sanMove}`);
				return sanMove;
			}
		} catch {
			// Invalid move, continue searching
		}
	}

	for (let i = mid + 1; i < data.length; i++) {
		const sanMove = tokenizer.decode([data[i]]).trim();
		try {
			const testChess = new Chess(chess.fen());
			const move = testChess.move(sanMove);
			if (move) {
				console.log(`Found valid move: ${sanMove}`);
				return sanMove;
			}
		} catch {
			// Invalid move, continue searching
		}
	}

	// Fallback: return first legal move if model fails
	const legalMoves = chess.moves();
	if (legalMoves.length > 0) {
		console.log(`Model failed, using random move: ${legalMoves[0]}`);
		return legalMoves[0];
	}

	return "";
}