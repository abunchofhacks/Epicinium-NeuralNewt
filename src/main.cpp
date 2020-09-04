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
 * Sander in 't Veld (sander@abunchofhacks.coop)
 */

#include <iostream>
#include <unordered_map>
#include <chrono>
#ifdef _MSC_VER
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#include "libs/jsoncpp/json.h"
#include "libs/aftermath/writer.hpp"
#include "libs/aftermath/library.hpp"
#include "libs/aftermath/loginstaller.hpp"

#include "setting.hpp"
#include "newtbraintrainer.hpp"


static std::unordered_map<std::string, Setting> settings =
	Setting::readSettings("settings.json");

static uint64_t currentMilliseconds()
{
	auto now = std::chrono::system_clock::now().time_since_epoch();
	return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}

void run(int argc, char* argv[])
{
	Writer writer;
	writer.install();
	Library library;
	library.load();
	library.install();

	LogInstaller("main", 20, settings["aftermath_loglevel"]).install();

	if (argc == 2)
	{
		throw std::runtime_error(
			"Only one argument supplied, I do not know what to do with that");
	}
	if (argc > 4)
	{
		throw std::runtime_error("More than three arguments supplied, I do not"
			" know what to do with that");
	}
	std::string session;
	size_t round = 0;
	bool initEvolve = false;
	if (argc > 2)
	{
		if (argc == 4)
		{
			std::string third(argv[3]);
			if (third != "e") throw std::runtime_error("Third argument should"
				" be \"e\" to enable initial evolution. I do not know what to"
				" do with \"" + third + "\"");
			else initEvolve = true;
		}
		session = argv[1];
		std::string folder = "brains/" + session;
		struct stat buffer;
		if (stat(folder.c_str(), &buffer) != 0)
		{
			throw std::runtime_error("Directory " + folder
				+ " does not exist! For resuming, first argument should be an"
				  " existing folder in the brains directory");
		}
		try
		{
			round = std::stoul(argv[2]);
		}
		catch (...)
		{
			throw std::runtime_error("Second argument is not an integer! For"
				" resuming, second argument should be an existing round number"
				" for the given brains subdirectory");
		}
		std::string resumeFile = folder + "/round" + std::to_string(round)
			+ ".txt";
		if (stat(resumeFile.c_str(), &buffer) != 0)
		{
			throw std::runtime_error("File " + resumeFile
				+ " does not exist! For resuming, second argument should be an"
				  " existing round number for the given brains subdirectory");
		}
	}
	// else argc == 1 and session is not set, so no resuming

	srand(currentMilliseconds());

	NewtBrainTrainer trainer(settings, Library::nameCurrentBible());
	if (!session.empty()) trainer.resume(session, round, initEvolve);
	trainer.train();
}

int main(int argc, char* argv[])
{
	try
	{
		run(argc, argv);
	}
	catch (const std::exception& e)
	{
		std::cerr << "Exception: " << e.what() << std::endl;
		throw e;
	}
	catch (...)
	{
		std::cerr << "Unknown error" << std::endl;
		throw std::runtime_error("Unknown error");
	}
	return 0;
}
