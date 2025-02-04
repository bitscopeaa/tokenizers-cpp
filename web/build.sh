#!/bin/bash
set -euxo pipefail

rustup target add wasm32-unknown-emscripten

mkdir -p build
cd build
emcmake cmake ../.. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3"
emmake make tokenizers_cpp tokenizers_c sentencepiece-static -j8
cd ..

emcc -o  src/tokenizers_binding.js src/tokenizers_binding.cc\
  build/libtokenizers_cpp.a build/libtokenizers_c.a   build/sentencepiece/src/libsentencepiece.a\
 -O3  -s ERROR_ON_UNDEFINED_SYMBOLS=0 -s EXPORT_ES6=1 -s EXPORTED_FUNCTIONS="_malloc,_free,_FromBlobJSON,_Encode,_Decode,_FromBlobByteLevelBPE,_FromBlobSentencePiece" -s MODULARIZE=1 -s SINGLE_FILE=1 -s EXPORTED_RUNTIME_METHODS=FS,getValue,stringToUTF8,UTF8ToString -s ALLOW_MEMORY_GROWTH=1\
 -I../include
