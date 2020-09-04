/**
 * Part of Epicinium NeuralNewt
 * developed by A Bunch of Hacks.
 *
 * Copyright (c) 2020 A Bunch of Hacks
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * [authors:]
 * Daan Mulder (daan@abunchofhacks.coop)
 */

#pragma once

#include "gamedirector.hpp"

#include "libs/aftermath/aihungryhippo.hpp"
#include "libs/aftermath/aiquickquack.hpp"
#include "libs/aftermath/airampantrhino.hpp"


template class GameDirector<AIHungryHippo, AIQuickQuack, AIRampantRhino>;
template void GameDirector<AIHungryHippo, AIQuickQuack, AIRampantRhino>::
	addAIGame<AIHungryHippo>(size_t, bool);
template void GameDirector<AIHungryHippo, AIQuickQuack, AIRampantRhino>::
	addAIGame<AIQuickQuack>(size_t, bool);
template void GameDirector<AIHungryHippo, AIQuickQuack, AIRampantRhino>::
	addAIGame<AIRampantRhino>(size_t, bool);
