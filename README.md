# Epicinium NeuralNewt

Framework for training neural networks ("brains")
to play [Epicinium](https://epicinium.nl)
via [*AINeuralNewt*](https://github.com/abunchofhacks/Epicinium-lib-rs/blob/master/epicinium/src/ai/aineuralnewt.cpp),
a parameterized decision tree AI,
with evolutionary training techniques.

The brains take as input
a three-dimensional representation of an Epicinium board state,
and provide as output a set of parameters for AINeuralNewt.
A population of brains is randomly initialized
(or loaded from a previous training session),
and improved through *selection*, *mutation* and *recombination*
in a configurable number of rounds.
In each round, the brains play games of Epicinium
against each other in a round-robin tournament,
supplemented with games against the existing (pure decision-tree) AIs,
HungryHippo, QuickQuack and RampantRhino, as a baseline.
Each game grants a score between 0 and 100.
The brains with the best total score
are selected (let through as-is),
mutated (a normally distributed number is
added to randomly selected weights and biases)
and recombined (two brains produce children
that each have the opposite random selection
of weights and biases from both parents)
to form the population for the next round.

In this way, brains are evolved to provide
increasingly better parameters for board states,
corresponding to better move outputs in AINeuralNewt.
This allows for flexible creation and improvement of AI
for current and future versions of Epicinium.
In our training sessions with non-dedicated hardware,
top brains on average outperformed HungryHippo and QuickQuack,
while being roughly equally matched with RampantRhino,
after about 200 hours ≈ 800 rounds of training.

You can configure any number of pools,
within which brains will only procreate with each other,
even though each brain is still pitched against
all others in the round-robin tournament.
The idea was that we might get some rock-paper-scissors thing going between the pools,
but this has not yet been observed in practice.

The current network architectures has
four convolutional layers followed by two fully connected layers.
We have no reason to believe this is optimal,
more experiments are needed!

[AINeuralNewt's source](https://github.com/abunchofhacks/Epicinium-lib-rs/blob/master/epicinium/src/ai/aineuralnewt.cpp) can be found in the *Epicinium-lib-rs* repository,
and is implemented in the Epicinium-NeuralNewt-automatonlib dependency,
both available under the AGPL-3.0 license.

The rest of Epicinium will be open-sourced in the near future.
It will also be coming soon to [Steam](https://epicinium.nl/steam).

Parts of the code are inspired by
[Surag Nair's helpful PyTorch implementation based on AlphaZero](https://github.com/suragnair/alpha-zero-general),
distributed under the MIT License.

Epicinium is being developed by [A Bunch of Hacks](https://abunchofhacks.coop),
a worker cooperative for video game and software development from the Netherlands.
Contact us at [info@epicinium.nl](mailto:info@epicinium.nl).

## Contents

* In `src/`, headers and source for Epicinium NeuralNewt
* `brains/`, `logs/` and `recordings/`, output folders.
* `settings.json`, an example configuration for the training algorithm.
* In `libs/jsoncpp/`, [*JsonCpp*](https://github.com/open-source-parsers/jsoncpp), a dependency of Epicinium NeuralNewt
* In `libs/openssl/`, headers and win64/debian64 lib for [*OpenSSL* 1.1.1a](https://github.com/openssl/openssl), a dependency of Epicinium NeuralNewt (only used for SHA256 cryptography for brain names)

## External dependencies

* *libtorch* 1.4.0 for CUDA 10.1 (other versions untested) ([Linux](https://download.pytorch.org/libtorch/cu101/libtorch-shared-with-deps-1.4.0.zip) or [Windows](https://download.pytorch.org/libtorch/cu101/libtorch-win-shared-with-deps-1.4.0.zip))
* [*Epicinium NeuralNewt automaton library*](https://github.com/abunchofhacks/Epicinium-NeuralNewt-automatonlib) (fetch with `git submodule update --init --recursive`)
* [*CMake*](https://cmake.org/download/), at least 3.7

## Compilation

### Linux

1. Install libtorch 1.4.0.
2. Make sure that CMake can find libtorch:
  - `TORCHCONFIG=[absolute path to TorchConfig.cmake]`
  - `mkdir cmake`
  - `echo "include($TORCHCONFIG)" > cmake/FindTorch.cmake`
3. Fetch the automatonlib: `git submodule update --init --recursive`
4. Run CMake from a separate build directory:
  - `mkdir build`
  - `cd build`
  - `cmake ..`
5. Execute the Makefile in the build directory:
`make`
6. Change `settings.json` as desired and run `./main`

`CMakeLists.txt` also defines a build target `neuralnewt` that compiles *libneuralnewt*, which is used in Epicinium to run NeuralNewt brains.

### Windows
Similar to above, but for step 5, we used CMake to produce a Visual Studio 14 project file: `cmake -G "Visual Studio 14 2015 Win64" ..`.

## License

*Epicinium NeuralNewt* was created by [A Bunch of Hacks](https://abunchofhacks.coop).
It is made available to you under the Apache License 2.0,
as specified in `LICENSE.txt`.

*JsonCpp* was created by Baptiste Lepilleur
and released under the MIT License,
or public domain wherever recognized,
as specified in `libs/jsoncpp/LICENSE`.

*OpenSSL* was created by The OpenSSL Project
and released under the dual OpenSSL and SSLeay license,
as specified in `libs/openssl/LICENSE`.

## Related repositories

*  [Epicinium documentation](https://github.com/abunchofhacks/epicinium-documentation), which includes a wiki and a tutorial for Epicinium
*  [Epicinium-NeuralNewt-automatonlib](https://github.com/abunchofhacks/Epicinium-NeuralNewt-automatonlib), a precompiled C++ library with Epicinium logic for Epicinium-NeuralNewt
*  [Epicinium-lib-rs](https://github.com/abunchofhacks/Epicinium-NeuralNewt-automatonlib), Rust bindings for libepicinium, the game logic library of Epicinium