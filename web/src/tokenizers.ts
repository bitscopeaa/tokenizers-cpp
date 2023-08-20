import Module from "./tokenizers_binding"

let binding: any = null;

async function asyncInitTokenizers() {
  if (binding == null) {
    binding = await Module();
  }
}

/**
 * A universal tokenizer that is backed by either
 * HF tokenizers rust library or sentencepiece
 */
export class Tokenizer {
  private handle: any;

  private constructor(tokenizer) {
    this.handle = tokenizer;
  }

  /**
   * Dispose this tokenizer.
   *
   * Call this function when we no longer needs to
   */
  dispose() {
      binding._free(this.handle);
  }

  /**
   * Encode text to token ids.
   *
   * @param text Input text.
   * @returns The output tokens
   */
  encode(text: string): Int32Array {
	  console.log("encodetext="+text);
	  const lengthPtr = binding._malloc(4);
	  const outputPtr = binding._malloc(text.length);
	  binding.stringToUTF8(text, outputPtr, text.length+1);
	  const ids = binding._Encode(this.handle, outputPtr, text.length, lengthPtr);
	  console.log("ids="+ids);
	  const arrayLength = binding.getValue(lengthPtr, "i32");
	  const intArray = new Int32Array(binding.HEAP32.buffer, ids, arrayLength);
	  binding._free(lengthPtr);
	  binding._free(outputPtr);
	  binding._free(ids);
	  return intArray;
  }

  /**
   * Decode the token ids into string.
   *
   * @param ids the input ids.
   * @returns The decoded string.
   */
  decode(ids: Int32Array): string {
    const lengthPtr = binding._malloc(4);
    const vec = binding._malloc(ids.byteLength);
    binding.HEAP32.set(ids, vec>>2);
    const res = binding._Decode(this.handle,vec, ids.length, lengthPtr);
    const arrayLength = binding.getValue(lengthPtr, "i32");
    const decodeResult= binding.UTF8ToString(res, arrayLength);
    binding._free(vec);
    binding._free(lengthPtr);
    return decodeResult;
  }

  /**
   * Create a tokenizer from jsonArrayBuffer
   *
   * @param json The input array buffer that contains json text.
   * @returns The tokenizer
   */
  static async fromJSON(json: ArrayBuffer): Promise<Tokenizer> {
    await asyncInitTokenizers();
    const jsonUint8Array = new Uint8Array(json);
    const arrayBufferPtr = binding._malloc(jsonUint8Array.length);
    binding.HEAPU8.set(new Uint8Array(json), arrayBufferPtr);

    const handle = new Tokenizer(binding._FromBlobJSON(arrayBufferPtr, jsonUint8Array.length));
    binding._free(arrayBufferPtr);
    return handle
  }

  /**
   * Create a tokenizer from byte-level BPE blobs.
   *
   * @param vocab The vocab blob.
   * @param merges The merges blob.
   * @param addedTokens The addedTokens blob
   * @returns The tokenizer
   */
  static async fromByteLevelBPE(
      vocab: ArrayBuffer,
      merges: ArrayBuffer,
      addedTokens = ""
  ) : Promise<Tokenizer> {
      await asyncInitTokenizers();

       const jsonUint8ArrayVocab = new Uint8Array(vocab);
       const arrayBufferVocabPtr = binding._malloc(jsonUint8ArrayVocab.length);
       binding.HEAPU8.set(new Uint8Array(vocab), arrayBufferVocabPtr);

       const jsonUint8ArrayMerges = new Uint8Array(merges);
       const arrayBufferMergesPtr = binding._malloc(jsonUint8ArrayMerges.length);
       binding.HEAPU8.set(new Uint8Array(merges), arrayBufferMergesPtr);
       const outputPtr = binding._malloc(4);
       binding.stringToUTF8(addedTokens, outputPtr, addedTokens.length+1);


      const  handle = new Tokenizer(
        binding._FromBlobByteLevelBPE(jsonUint8ArrayVocab, jsonUint8ArrayVocab.length, merges, outputPtr));
	binding._free(arrayBufferVocabPtr);
	binding._free(arrayBufferMergesPtr);
	binding._free(outputPtr);
	return handle;
  }

  /**
   * Create a tokenizer from sentencepiece model.
   *
   * @param model The model blob.
   * @returns The tokenizer
   */
   static async fromSentencePiece(model: ArrayBuffer) : Promise<Tokenizer> {
    await asyncInitTokenizers();
       const jsonUint8Array= new Uint8Array(model);
       const arrayBufferPtr = binding._malloc(jsonUint8Array.length);
       console.log("modellen="+jsonUint8Array.length);
       binding.HEAPU8.set(new Uint8Array(model), arrayBufferPtr);
       const handle = new Tokenizer(
        binding._FromBlobSentencePiece(arrayBufferPtr, jsonUint8Array.length ));
	binding._free(arrayBufferPtr);
	return handle;
  }
}
