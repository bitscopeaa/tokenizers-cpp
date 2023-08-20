#include <emscripten.h>
#include <emscripten/bind.h>
#include <tokenizers_cpp.h>
#include <stdio.h>

#include <memory>
#include <iostream>

extern "C" {
	 EMSCRIPTEN_KEEPALIVE
	 tokenizers::Tokenizer*  FromBlobJSON(const char* data, int datalen){
		 std::string jsondata(data, datalen);
		 std::unique_ptr<tokenizers::Tokenizer> tokenizer =   tokenizers::Tokenizer::FromBlobJSON(jsondata); 
		 tokenizers::Tokenizer* point = tokenizer.get();
		 tokenizer.release();
		 return point;
	 }

         EMSCRIPTEN_KEEPALIVE
         tokenizers::Tokenizer*  FromBlobByteLevelBPE(
			 const char* vocab_char, 
			 const int vocab_length,
                         const char* merges_char,
			 const int merges_length,
                         const char* added_tokens_char, 
			 const int added_tokens_length){
		  std::string vocab(vocab_char, vocab_length);
		  std::string merges(merges_char, merges_length);
		  std::string added_tokens(added_tokens_char, added_tokens_length);
                   std::unique_ptr<tokenizers::Tokenizer> tokenizer =   tokenizers::Tokenizer::FromBlobByteLevelBPE(vocab, merges, added_tokens); 
                   tokenizers::Tokenizer* point = tokenizer.get();
                   tokenizer.release();
                   return point;
         }

	 EMSCRIPTEN_KEEPALIVE
         tokenizers::Tokenizer*  FromBlobSentencePiece(const char* model_blob_char, const int model_blob_length){
		 std::string model_blob(model_blob_char, model_blob_length);
                   std::unique_ptr<tokenizers::Tokenizer> tokenizer =   tokenizers::Tokenizer::FromBlobSentencePiece(model_blob); 
                   tokenizers::Tokenizer* point = tokenizer.get();
                   tokenizer.release();
                   return point;
         }

	 EMSCRIPTEN_KEEPALIVE
         int* Encode(tokenizers::Tokenizer* tokenizer, const char* text, const int text_len, int *output_length){
                   std::vector<int32_t> data =   tokenizer->Encode(std::string(text, text_len)); 
                   int* arr = new int[data.size()];
                   for (size_t i = 0; i < data.size(); ++i) {
                           arr[i] = data[i];
                   }
		   *output_length = data.size();
                   return arr;
         }
         
        EMSCRIPTEN_KEEPALIVE
        const char* Decode(tokenizers::Tokenizer* tokenizer,  const int* ids, const int ids_length, int* output_length) {
	      std::vector<int32_t> vectorIds;
	      for(int i=0; i<ids_length; ++i){
		      vectorIds.push_back(ids[i]);
	      }
              std::string data = tokenizer->Decode(vectorIds);
	      *output_length = data.size();
	      return data.c_str();
        }
}
