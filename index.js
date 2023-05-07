import { Tokenizer } from "huggingface-tokenizers-bindings";

const INPUT = "As a surveyor, I want to be able to log into a system";

async function main() {
    let tokenizer = await Tokenizer.from_pretrained("bert-base-cased");
    let encoding = tokenizer.encode(INPUT, false);
    document.getElementById("input").innerHTML = INPUT;
    document.getElementById("tokens").innerHTML = "[" + encoding.tokens + "]"
    document.getElementById("input_ids").innerHTML = "[" + encoding.input_ids + "]";
}

main();