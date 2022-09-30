/**
 * Framework for Threes! and its variants (C++ 11)
 * agent.h: Define the behavior of variants of agents including players and environments
 *
 * Author: Theory of Computer Games
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <iostream>
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include <fstream>
#include <climits>
#include "board.h"
#include "action.h"
#include "weight.h"

using namespace std;

class agent
{
public:
	agent(const std::string &args = "")
	{
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair;)
		{
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = {value};
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string &flag = "") {}
	virtual void close_episode(const std::string &flag = "") {}
	virtual action take_action(const board &b) { return action(); }
	virtual bool check_for_win(const board &b) { return false; }

public:
	virtual std::string property(const std::string &key) const { return meta.at(key); }
	virtual void notify(const std::string &msg) { meta[msg.substr(0, msg.find('='))] = {msg.substr(msg.find('=') + 1)}; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

protected:
	typedef std::string key;
	struct value
	{
		std::string value;
		operator std::string() const { return value; }
		template <typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

/**
 * base agent for agents with randomness
 */
class random_agent : public agent
{
public:
	random_agent(const std::string &args = "") : agent(args)
	{
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * base agent for agents with weight tables and a learning rate
 */
class weight_agent : public agent
{
public:
	weight_agent(const std::string &args = "") : agent(args), alpha(0)
	{
		if (meta.find("init") != meta.end())
			init_weights(meta["init"]);
		if (meta.find("load") != meta.end())
			load_weights(meta["load"]);
		if (meta.find("alpha") != meta.end())
			alpha = float(meta["alpha"]);
	}
	virtual ~weight_agent()
	{
		if (meta.find("save") != meta.end())
			save_weights(meta["save"]);
	}

protected:
	virtual void init_weights(const std::string &info)
	{
		std::string res = info; // comma-separated sizes, e.g., "65536,65536"
		for (char &ch : res)
			if (!std::isdigit(ch))
				ch = ' ';
		std::stringstream in(res);
		for (size_t size; in >> size; net.emplace_back(size))
			;
	}
	virtual void load_weights(const std::string &path)
	{
		std::ifstream in(path, std::ios::in | std::ios::binary);
		if (!in.is_open())
			std::exit(-1);
		uint32_t size;
		in.read(reinterpret_cast<char *>(&size), sizeof(size));
		net.resize(size);
		for (weight &w : net)
			in >> w;
		in.close();
	}
	virtual void save_weights(const std::string &path)
	{
		std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
		if (!out.is_open())
			std::exit(-1);
		uint32_t size = net.size();
		out.write(reinterpret_cast<char *>(&size), sizeof(size));
		for (weight &w : net)
			out << w;
		out.close();
	}

protected:
	std::vector<weight> net;
	float alpha;
};

/**
 * default random environment, i.e., placer
 * place the hint tile and decide a new hint tile
 */
class random_placer : public random_agent
{
public:
	random_placer(const std::string &args = "") : random_agent("name=place role=placer " + args)
	{
		spaces[0] = {12, 13, 14, 15};										// buttom
		spaces[1] = {0, 4, 8, 12};											// left
		spaces[2] = {0, 1, 2, 3};											// up
		spaces[3] = {3, 7, 11, 15};											// right
		spaces[4] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}; // all
	}

	virtual action take_action(const board &after)
	{
		std::vector<int> space = spaces[after.last()];
		std::shuffle(space.begin(), space.end(), engine);
		for (int pos : space)
		{
			if (after(pos) != 0)
				continue;

			int bag[3], num = 0;
			for (board::cell t = 1; t <= 3; t++)
				for (size_t i = 0; i < after.bag(t); i++)
					bag[num++] = t;
			std::shuffle(bag, bag + num, engine);

			board::cell tile = after.hint() ?: bag[--num];
			board::cell hint = bag[--num];

			return action::place(pos, tile, hint);
		}
		return action();
	}

private:
	std::vector<int> spaces[5];
};

/**
 * random player, i.e., slider
 * select a legal action randomly
 */
class random_slider : public random_agent
{
public:
	random_slider(const std::string &args = "") : random_agent("name=slide role=slider " + args),
												  opcode({0, 1, 2, 3}) {}

	virtual action take_action(const board &before)
	{
		std::shuffle(opcode.begin(), opcode.end(), engine);
		for (int op : opcode)
		{
			board::reward reward = board(before).slide(op);
			if (reward != -1)
				return action::slide(op);
		}
		return action();
	}

private:
	std::array<int, 4> opcode;
};

/**
 * greedy player, i.e., slider
 * select a legal action by greedy alg.
 */
class greedy_slider : public random_agent
{
public:
	greedy_slider(const std::string &args = "") : random_agent("name=greedy_slider role=slider " + args) {}
	virtual action take_action(const board &before)
	{
		board::reward maxReward = INT_MIN;
		int index = 0;

		for(int i = 0; i < 4; i++){
			board::reward rwd = board(before).slide(i);
			if(maxReward < rwd){
				maxReward = rwd;
				index = i;
			}
		}
		if(maxReward != -1) {
			return action::slide(index);
		}

		return action();
	}
};

/**
 * two step greedy player, i.e., slider
 * select a legal action by greedy alg.
 */
class two_step_greedy_slider : public random_agent
{
public:
	two_step_greedy_slider(const std::string &args = "") : random_agent("name=two_step_greedy_slider role=slider " + args) {}

	virtual action take_action(const board &before)
	{
		board::reward maxFirstReward = INT_MIN;
		int index = 0;

		for(int i = 0; i < 4; i++){
			board board1 = board(before);
			board::reward FirstReward = board1.slide(i);
			board::reward maxSecondReward = INT_MIN;	

			for(int j = 0; j < 4; j++){
				board board2 = board(board1);
				maxSecondReward = max(maxSecondReward, board2.slide(j));
			}

			if(maxSecondReward + FirstReward > maxFirstReward){
				maxFirstReward = maxSecondReward + FirstReward;
				index = i;
			}
		}
		if(maxFirstReward != -1) {
			return action::slide(index);
		}

		return action();
	}
};


