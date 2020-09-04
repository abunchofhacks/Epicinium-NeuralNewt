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

#include <memory>
#include <string>

class BrainName;
class BrainNamePtr : public std::shared_ptr<BrainName>
{
	using std::shared_ptr<BrainName>::shared_ptr;
};


class BrainName
{
private:
	std::string _mediumName;
	std::string _shortName;
	size_t _round;

	friend class SeedBrainName;
	friend class MuBrainName;
	friend class CoBrainName;
	friend class RestoredBrainName;

	BrainName(const std::string& mediumName, const std::string& shortName,
		size_t round);
	virtual ~BrainName() = default;

	virtual std::string shortenedName(size_t depth) const = 0;

	virtual std::string longName() const = 0;

public:
	std::string mediumName() const { return _mediumName; }
	std::string shortName() const { return _shortName; }
};

class SeedBrainName : public BrainName
{
public:
	SeedBrainName(size_t n);

private:
	std::string shortenedName(size_t depth) const override;

	std::string longName() const override;
};

class MuBrainName : public BrainName
{
private:
	BrainNamePtr _parent;

public:
	MuBrainName(const BrainNamePtr& parent, size_t round);

private:
	std::string shortenedName(size_t depth) const override;

	std::string longName() const override;
};

class CoBrainName : public BrainName
{
private:
	BrainNamePtr _parent1, _parent2;

public:
	CoBrainName(const BrainNamePtr& parent1, const BrainNamePtr& parent2,
		size_t round);

private:
	std::string shortenedName(size_t depth) const override;

	std::string longName() const override;
};

class RestoredBrainName : public BrainName
{
public:
	RestoredBrainName(const std::string& name, size_t round);

private:
	std::string shortenedName(size_t depth) const override;

	std::string longName() const override;
};
