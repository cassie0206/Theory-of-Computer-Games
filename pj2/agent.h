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
struct state {
	board before;
	board after;
	int reward;
	float value;
    bool isSlider = false;
};

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b, state &s) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

/**
 * base agent for agents with randomness
 */
class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
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
class weight_agent : public agent {
public:
	weight_agent(const std::string& args = "") : agent(args), alpha(0) {
        if (meta.find("alpha") != meta.end())
			alpha = float(meta["alpha"]);
		if (meta.find("step") != meta.end())
			step = int(meta["step"]);
	}

protected:
	virtual void init_weights(const std::string& info) {}
	virtual void load_weights(const std::string& path) {
		std::ifstream in(path, std::ios::in | std::ios::binary);
		if (!in.is_open()) std::exit(-1);
		uint32_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(size));
		net.resize(size);
		for (weight& w : net) in >> w;
		in.close();
	}
	virtual void save_weights(const std::string& path) {
		std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
		if (!out.is_open()) std::exit(-1);
		uint32_t size = net.size();
		out.write(reinterpret_cast<char*>(&size), sizeof(size));
		for (weight& w : net) out << w;
		out.close();
	}

protected:
	std::vector<weight> net;
	float alpha;
	int step;
};

/**
 * default random environment, i.e., placer
 * place the hint tile and decide a new hint tile
 */
class random_placer : public random_agent {
public:
	random_placer(const std::string& args = "") : random_agent("name=place role=placer " + args) {
		spaces[0] = { 12, 13, 14, 15 };
		spaces[1] = { 0, 4, 8, 12 };
		spaces[2] = { 0, 1, 2, 3};
		spaces[3] = { 3, 7, 11, 15 };
		spaces[4] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
	}

	virtual action take_action(const board& after, state &s) {
		std::vector<int> space = spaces[after.last()];
		std::shuffle(space.begin(), space.end(), engine);
		for (int pos : space) {
			if (after(pos) != 0) continue;

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
class random_slider : public random_agent {
public:
	random_slider(const std::string& args = "") : random_agent("name=slide role=slider " + args),
		opcode({ 0, 1, 2, 3 }) {}

	virtual action take_action(const board& before, state &s) {
		std::shuffle(opcode.begin(), opcode.end(), engine);
		for (int op : opcode) {
			board::reward reward = board(before).slide(op);
			if (reward != -1) return action::slide(op);
		}
		return action();
	}

private:
	std::array<int, 4> opcode;
};
class TDL_slider : public weight_agent {
public:
	TDL_slider(const std::string& args = "") : weight_agent(args), opcode({ 0, 1, 2, 3 }), space({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }) {
		for(int i=0;i<4;i++){
			net.emplace_back(weight(16 * 16 * 16 * 16 * 16 * 16));
		}
		if (meta.find("load") != meta.end())
			load_weights(meta["load"]);

        spaces[0] = { 12, 13, 14, 15 };
		spaces[1] = { 0, 4, 8, 12 };
		spaces[2] = { 0, 1, 2, 3};
		spaces[3] = { 3, 7, 11, 15 };
		cout<<"number of step: "<<step<<"\n";
	}
    virtual ~TDL_slider()
	{
		if (meta.find("save") != meta.end())
			save_weights(meta["save"]);
	}

	virtual action take_action(const board& before, state &s) 
    {   
        s.isSlider = true;
        float best_value = -numeric_limits<float>::max();
		float best_cur_val = -numeric_limits<float>::max();
		int best_reward = -1;
		int best_op = -1; 
		for(int i=0;i<4;i++){
			board tmp = board(before);
			board::reward reward = tmp.slide(i);
			if (reward == -1) {
				continue;
			}
			float value = expectimax(tmp, i);	
						
			if (value + reward > best_value) {
				best_value = value + reward;
				best_op = i;
				best_reward = reward;
				best_cur_val = value;
			}

		}

        if(best_op != -1){
			//state_value = best_cur_val;
			//r = best_reward;
            s.reward = best_reward;
            s.value = best_cur_val;
			return action::slide(best_op);
		}
		else{
			return action();
		}
		
	}
	
	float expectimax(const board &b, int op){
		std::vector<int> unoccupied;
		std::vector<int> space = spaces[op];
		for(int i=0;i<4;i++){
			if(b(space[i]) == 0){
				unoccupied.push_back(space[i]);
			} 
		}
		int total = unoccupied.size();

		int bag[3], num = 0;
		for (board::cell t = 1; t <= 3; t++)
			for (size_t i = 0; i < b.bag(t); i++)
				bag[num++] = t;

		std::shuffle(bag, bag + num, engine);
		board::cell tile = b.hint();
		board::cell hint = bag[--num];
		float sum = 0.0;

		for(int i : unoccupied){
			board tmp = board(b);
			tmp.place(i, tile, hint);
			board::reward best_reward = -1;
			float best_val = -std::numeric_limits<float>::max();

			for(int j=0;j<4;j++){
				board tmp_after = board(tmp);
				board::reward reward = tmp_after.slide(j);
				if(reward == -1) continue;
				
				float val = get_value(tmp_after);
				if(reward + val > best_reward + best_val) {
					best_reward = reward;
					best_val = val;
				}
			}

			if(best_reward == -1){
				continue;
			}		
			sum += (best_val + best_reward) / float(total);
		}

		return sum;
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

    int encode6(const board& board, int a, int b, int c, int d, int e, int f){
        return (board(a) << 20) | (board(b) << 16) | (board(c) << 12) | (board(d) << 8) | (board(e) << 4) | (board(f) << 0);
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

	void update_value(std::vector<state>& v) {
		float final_alpha = alpha / 32;
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
			adjust_weight(v[i].after, final_alpha * error);
		}
	}
private:
	std::array<int, 4> opcode;
	std::vector<int> space;
    std::default_random_engine engine;
    std::vector<int> spaces[4];
};

