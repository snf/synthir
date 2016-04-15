# Synthir

[![Build Status](https://travis-ci.org/snf/synthir.svg?branch=master)](https://travis-ci.org/snf/synthir)

This tool uses templates, MCMC and a mixed approach to synthetizes IR transformations for assembly instructions (currently only supporting x86_64).

Slides from Ekoparty: [IR transformation synthesis for assembly instructions](https://speakerdeck.com/snf/ir-transformation-synthesis-for-assembly-instructions)


## Installation

Runs with Rust >= 1.7 and requires the following libraries:

* libz-dev
* libedit-dev
* libcapstone-dev
* llvm-3.8-dev (or 3.7)

## More information

* Automated Synthesis of Symbolic Instruction Encodings from I/O Samples: [Paper](http://research.microsoft.com/en-us/um/people/pg/public_psfiles/pldi2012.pdf)
* Stochastic Superoptimization: [Paper](http://cs.stanford.edu/people/sharmar/pubs/asplos291-schkufza.pdf)

## TODO

- [x] x86_64 support
- [ ] ARM support (synthir_execute)
- [ ] Mips support (synthir_execute)
- [ ] Finish Floating point support (SMT and emulator)
- [ ] Add back mixed approach
- [ ] Extend available templates
- [ ] Transform our IR to LLVM (synthir_llvm)
- [ ] Clean
