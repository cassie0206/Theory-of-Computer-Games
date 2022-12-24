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
#include <limits>
#include <stack>
#include "board.h"
#include "action.h"
#include "weight.h"

using namespace std;

struct state{
	board before;
	board after;
	float value = 0.0;
	int reward = 0;
	bool isSlider = false;
};

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
	virtual action take_action(const board &b, state &s) { return action(); }
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
		/*if (meta.find("init") != meta.end())
			init_weights(meta["init"]);
		if (meta.find("load") != meta.end())
			load_weights(meta["load"]);*/
		if (meta.find("alpha") != meta.end())
			alpha = float(meta["alpha"]);
		if (meta.find("step") != meta.end())
			step = int(meta["step"]);
	}
	/*virtual ~weight_agent()
	{
		if (meta.find("save") != meta.end())
			save_weights(meta["save"]);
	}*/

protected:
	virtual void init_weights(const std::string &info)
	{
		/*std::string res = info; // comma-separated sizes, e.g., "65536,65536"
		for (char &ch : res)
			if (!std::isdigit(ch))
				ch = ' ';
		std::stringstream in(res);
		for (size_t size; in >> size; net.emplace_back(size))
			;*/
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
	int step=1;
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

	virtual action take_action(const board &after, state &s)
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

	virtual action take_action(const board &before, state &s)
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

class TDL_slider : public weight_agent
{
public:
	TDL_slider(const std::string &args = "") : weight_agent("name=TDL_slider role=slider " + args) {
		for(int i=0;i<4;i++){
			net.emplace_back(weight(16 * 16 * 16 * 16 * 16 * 16));
		}
		if (meta.find("load") != meta.end())
			load_weights(meta["load"]);

		count = 0;
	}
	virtual ~TDL_slider()
	{
		if (meta.find("save") != meta.end())
			save_weights(meta["save"]);
	}
												 
	virtual action take_action(const board &before, state &s)
	{	
		s.isSlider = true;
		float best_value = -numeric_limits<float>::max();
		float best_cur_val = -numeric_limits<float>::max();
		int best_reward = -1;
		int best_op = -1; 
		for(int i=0;i<4;i++){
			board tmp = board(before);
			board::reward reward = tmp.slide(i);
			if(reward == -1){
				continue;
			}
			float value = expectimax(tmp);

			if(value + reward > best_value){
				best_value = value + reward;
				best_op = i;
				best_reward = reward;
				best_cur_val = value;
			}
		}

		if(best_op != -1 && best_reward != -1){
			s.reward = best_reward;
			s.value = best_cur_val;
			return action::slide(best_op);
		}
		else{
			return action();
		}
	}

	float get_value(const board &b){
		float val = 0.0;
		board tmp = board(b);
		// isomorphism * 8 (rotate + reflect)
		for(int i=0;i<2;i++){
			for(int j=0;j<4;j++){
				val += net[0][encode6(tmp, 0, 1, 2, 3, 4, 5)];
				val += net[1][encode6(tmp, 4, 5, 6, 7, 8, 9)];
				val += net[2][encode6(tmp, 5, 6, 7, 9, 10, 11)];
				val += net[3][encode6(tmp, 9, 10, 11, 13, 14, 15)];

				tmp.rotate_clockwise();
			}
			tmp.reflect_horizontal();
		}
		return val;
	}

	float expectimax(const board &b){
		vector<int> unoccupied;

		for(int i=0;i<16;i++){
			if(b(i) == 0){
				unoccupied.push_back(i);
			}
		}

		//initial tile may be 1, 2, 3
		float total_val = 0.0;
		int num = 0;
		for(board::cell i=1;i<4;i++){
			for(int pos:unoccupied){
				board tmp = b;
				tmp.set_tile(pos, i);
				float best_val = -numeric_limits<float>::max();
				num++;

				for(int j=0;j<4;j++){
					board tmp_after = tmp;
					board::reward reward = tmp_after.slide(j);
					if(reward == -1){
						continue;
					}

					float val = get_value(tmp_after);
					if(reward + val > best_val){
						best_val = reward + val;
					}
				}
				total_val += best_val;
			}
		}

		return (total_val / num);
	}

	int encode6(const board& board, int a, int b, int c, int d, int e, int f){
		return (board(a) << 0) | (board(b) << 4) | (board(c) << 8) | (board(d) << 12) | (board(e) << 16) | (board(f) << 20);
	}

	void adjust_weight(const board& b, float target){
		board tmp = board(b);

		for(int i=0;i<2;i++){
			for(int j=0;j<4;j++){
				net[0][encode6(tmp, 0, 1, 2, 3, 4, 5)] += target;
				net[1][encode6(tmp, 4, 5, 6, 7, 8, 9)] += target;
				net[2][encode6(tmp, 5, 6, 7, 9, 10, 11)] += target;
				net[3][encode6(tmp, 9, 10, 11, 13, 14, 15)] += target;

				tmp.rotate_clockwise();
			}
			tmp.reflect_horizontal();
		}
	}

	void update_value(vector<state> &v){
		for(int i = v.size() - 1 ; i >= 0 ; i--){
			board::reward total_reward = 0;
			float error;
			bool flag = false;
			for(int j = 1 ; j <= step ; j++){
				if(i + j >= int(v.size())){
					error = total_reward + 0 - get_value(v[i].after);
					flag = true;
					break;
				} 
				total_reward += v[i + j].reward;
			}

			if(!flag){
				error = total_reward + get_value(v[i + step].after) - get_value(v[i].after);
			}
			adjust_weight(v[i].after, alpha * error);
		}
	}

private:
	std::array<int, 4> opcode;
	int count;
};