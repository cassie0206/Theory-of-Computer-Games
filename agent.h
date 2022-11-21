/**
 * Framework for NoGo and similar games (C++ 11)
 * agent.h: Define the behavior of variants of the player
 *
 * Author: Theory of Computer Games
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <ctime>
#include "board.h"
#include "action.h"

using namespace std;

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
	virtual action take_action(const board& b) { return action(); }
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
		if (meta.find("timeout") != meta.end())
			timeout = (int(meta["seed"]));
		if (meta.find("simulation") != meta.end())
			simulation_time = (int(meta["simulation"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
	clock_t timeout = -1;
	int simulation_time = -1;
};

/**
 * random player for both side
 * put a legal piece randomly
 */
/*class player : public random_agent {
public:
	player(const std::string& args = "") : random_agent("name=random role=unknown " + args),
		space(board::size_x * board::size_y), who(board::empty) {
		if (name().find_first_of("[]():; ") != std::string::npos)
			throw std::invalid_argument("invalid name: " + name());
		if (role() == "black") who = board::black;
		if (role() == "white") who = board::white;
		if (who == board::empty)
			throw std::invalid_argument("invalid role: " + role());
		for (size_t i = 0; i < space.size(); i++)
			space[i] = action::place(i, who);
	}

	virtual action take_action(const board& state) {
		std::shuffle(space.begin(), space.end(), engine);
		for (const action::place& move : space) {
			board after = state;
			if (move.apply(after) == board::legal)
				return move;
		}
		return action();
	}

private:
	std::vector<action::place> space;
	board::piece_type who;
};
*/
/**
 * random player for both side
 * put a legal piece randomly
 */
class MCTS_player : public random_agent {
public:
	MCTS_player(const std::string& args = "") : random_agent("name=MCTS role=unknown " + args),
		space(board::size_x * board::size_y), black_space(board::size_x * board::size_y), white_space(board::size_x * board::size_y), who(board::empty) {
		if (name().find_first_of("[]():; ") != std::string::npos)
			throw std::invalid_argument("invalid name: " + name());
		if (role() == "black") who = board::black;
		if (role() == "white") who = board::white;
		if (who == board::empty)
			throw std::invalid_argument("invalid role: " + role());
		for (size_t i = 0; i < space.size(); i++)
			space[i] = action::place(i, who);
		for (size_t i = 0; i < space.size(); i++)
			black_space[i] = action::place(i, board::black);
		for (size_t i = 0; i < space.size(); i++)
			white_space[i] = action::place(i, board::white);
	}

	struct Node{
		int Tn = 0; // the visited time 
		int x = 0; // reward (# of win)
		Node* parent = NULL;
		vector<Node*> childs;
		board state;
		action::place parent_move;
		double UCT_val = 0;
		board::piece_type who;
	};

	void calculate_UCT(Node* n){
		double exploit = (double)n->x / (double)n->Tn;
		double explore= 0.5 * sqrt(((double) log(n->parent->Tn) / n->Tn));
		n->UCT_val = exploit + explore;
	}

	board::piece_type turn_who(board::piece_type who){
		if(who == board::black){
			return board::white;
		}
		else{
			return board::black;
		}
	}

	std::vector<action::place> which_space(board::piece_type who){
		if(who == board::black){
			return black_space;
		}
		else{
			return white_space;
		}
	}

	Node* selection(Node* root){
		Node* cur = root;
		board::piece_type curWho = who;

		while(cur->childs.size() != 0){
			double max_UCT = -1;
			int index = 0;
			for(int i=0;i<cur->childs.size();i++){
				if(cur->childs[i]->UCT_val > max_UCT){
					max_UCT = cur->childs[i]->UCT_val;
					index = i;
					curWho = turn_who(curWho);
					cur->childs[i]->who = curWho;
				}
			}
			cur = cur->childs[index];
		}

		return cur;
	}

	Node* expansion(Node* n){
		std::vector<action::place> cur_space = which_space(n->who);
		std::shuffle(space.begin(), space.end(), engine);
		for (const action::place& move : cur_space) {
			board after = n->state;
			if (move.apply(after) == board::legal){
				Node* child = new Node;
				child->parent = n;
				n->childs.push_back(child);
				child->who = turn_who(n->who);
				n->parent_move = move;
				n->Tn++;
				return child;
			}		
		}
	}

	int simulation(Node* n){
		Node* cur = n;
		board cur_state = n->state;
		while(1){
			bool flag = false;
			std::vector<action::place> cur_space = which_space(n->who);
			std::shuffle(space.begin(), space.end(), engine);
			for (const action::place& move : cur_space) {
				board after = cur_state;
				if (move.apply(after) == board::legal){
					cur_state = after;
					flag =true;
					break;
				}		
			}

			if(!flag){
				if(cur->who == who){//lose
					return 0;
				}
				else{//win
					return 1;
				}
			}
			cur = cur->childs[cur->childs.size() - 1];
		}
	}

	void backpropagation(Node* n, int re){
		Node* cur = n;//expansion node
		while(cur->parent != NULL){
			cur->Tn++;
			cur->x += re;
			calculate_UCT(cur);
			cur = cur->parent;
		}
	}

	void MCTS(Node* root){
		clock_t START_TIME, END_TIME;
		START_TIME = clock();
		int time = 0;

		while(1){
			Node* leaf = selection(root);
			Node* newNode = expansion(leaf);
			int re = simulation(newNode);
			backpropagation(newNode, re);
			time++;

			if(timeout != -1){
				END_TIME = clock();
				if(timeout > END_TIME - START_TIME){
					break;
				}
			}
			if(simulation_time != -1 && time == simulation_time){
				break;
			}
		}
	}

	virtual action take_action(const board& state) {
		Node* root;
		root->state = state;
		root->who;
		MCTS(root);

		double max_UCT;
		int index = 0;
		for(int i=0;i<root->childs.size();i++){
			if(root->childs[i]->UCT_val > max_UCT){
				root->childs[i]->UCT_val = max_UCT;
				index = i;
			}
		}

		return root->childs[index]->parent_move;
	}





private:
	std::vector<action::place> space;// next legal move place
	std::vector<action::place> black_space;
	std::vector<action::place> white_space;
	board::piece_type who;
};

