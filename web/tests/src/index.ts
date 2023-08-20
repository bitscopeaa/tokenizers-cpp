import { Tokenizer } from "@mlc-ai/web-tokenizers";

async function testJSONTokenizer() {
  console.log("JSON Tokenizer");
  const jsonBuffer = await (await
    fetch("https://huggingface.co/openai/clip-vit-large-patch14/raw/main/tokenizer.json")
  ).arrayBuffer();

  console.log("jsonBuffer");
  console.log(jsonBuffer);
  const tok = await Tokenizer.fromJSON(jsonBuffer);
  console.log("loadok");
  console.log(tok);
  const text = "What is the capital of Canada?";
  const ids = tok.encode(text);
  console.log("ids=" + ids)
  const decodedText = tok.decode(ids);
  console.log("decoded=" + decodedText);
}

async function testLlamaTokenizer() {
  console.log("Llama Tokenizer");
  const modelBuffer = await (await
      fetch("https://huggingface.co/hongyij/web-llm-test-model/resolve/main/tokenizer.model")
  ).arrayBuffer();
  console.log("len="+modelBuffer.length);
  const tok = await Tokenizer.fromSentencePiece(modelBuffer);
  console.log("tok:");
  console.log(tok);
  const text = "What is the capital of Canada?";
  const ids = tok.encode(text);
  console.log("ids=" + ids)
  const decodedText = tok.decode(ids);
  console.log("decoded=" + decodedText);
}

async function main() {
  await testJSONTokenizer()
  await testLlamaTokenizer()
}

main()
