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

#include "setting.hpp"

#include <fstream>

#include "libs/jsoncpp/json.h"


std::unordered_map<std::string, Setting> Setting::readSettings(
	const std::string& filename)
{
	std::unordered_map<std::string, Setting> result;
	Json::Reader reader;
	Json::Value root;
	std::ifstream file(filename);
	if (!file.is_open())
	{
		throw std::runtime_error("Cannot open settings file: " + filename);
	}
	if (!reader.parse(file, root) || !root.isObject())
	{
		throw std::runtime_error("Cannot read settings file: " + filename);
	}
	for (const std::string& name : root.getMemberNames())
	{
		const Json::Value& value = root[name];
		if (value.isBool())
		{
			result.emplace(name, value.asBool());
		}
		else if (value.isInt())
		{
			result.emplace(name, value.asInt());
		}
		else if (value.isDouble())
		{
			result.emplace(name, value.asFloat());
		}
		else if (value.isString())
		{
			result.emplace(name, value.asString());
		}
		else if (value.isArray())
		{
			std::vector<std::string> v;
			for (const auto& x : value)
			{
				v.push_back(x.asString());
			}
			result.emplace(name, v);
		}
		else
		{
			throw std::runtime_error("Invalid value " + value.toStyledString()
				+ " in settings file: " + filename);
		}
	}
	return result;
}
