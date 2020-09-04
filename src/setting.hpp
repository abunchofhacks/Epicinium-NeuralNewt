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

#include <string>
#include <cassert>
#include <unordered_map>
#include <vector>


class Setting
{
public:
	static std::unordered_map<std::string, Setting> readSettings(
		const std::string& filename);

private:
	enum class Type : uint8_t
	{
		NONE,
		BOOL,
		INT,
		FLOAT,
		STRING,
		VECTOR,
	};

	union
	{
		bool _bValue;
		int _iValue;
		float _fValue;
	};
	// TODO in union?
	std::string _sValue;
	std::vector<std::string> _vValue;

	Type _type;

public:
	Setting() :
		_type(Type::NONE)
	{}

	Setting(bool value) :
		_bValue(value),
		_type(Type::BOOL)
	{}

	Setting(int value) :
		_iValue(value),
		_type(Type::INT)
	{}

	Setting(float value) :
		_fValue(value),
		_type(Type::FLOAT)
	{}

	Setting(const std::string& value) :
		_sValue(value),
		_type(Type::STRING)
	{}
	Setting(const char* value) :
		Setting(std::string(value))
	{}

	Setting(const std::vector<std::string>& value) :
		_vValue(std::move(value)),
		_type(Type::VECTOR)
	{}

	~Setting() = default;
	Setting(const Setting& that) = default;
	Setting(Setting&& that) = default;
	Setting& operator=(const Setting& that) = default;
	Setting& operator=(Setting&& that) = default;

	operator bool() const
	{
		assert(_type == Type::BOOL);
		return _bValue;
	}

	operator int() const
	{
		assert(_type == Type::INT);
		return _iValue;
	}
	operator size_t() const { return operator int(); }
	operator int64_t() const { return operator int(); }

	operator float() const
	{
		if (_type == Type::INT) return _iValue;
		assert(_type == Type::FLOAT);
		return _fValue;
	}

	operator std::string() const
	{
		assert(_type == Type::STRING);
		return _sValue;
	}

	operator std::vector<std::string>() const
	{
		assert(_type == Type::VECTOR);
		return _vValue;
	}

	std::string operator[](size_t i) const
	{
		assert(_type == Type::VECTOR);
		return _vValue[i];
	}

	size_t size() const
	{
		assert(_type == Type::VECTOR);
		return _vValue.size();
	}
};
