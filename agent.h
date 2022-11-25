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
	int simulation_time = 100;
};

/**
 * random player for both side
 * put a legal piece randomly
 */
class player : public random_agent {
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
			black_space[i] = action::black(i);
		for (size_t i = 0; i < space.size(); i++)
			white_space[i] = action::white(i);
	}

	struct Node{
		int Tn = 0; // the visited time 
		int x = 0; // reward (# of win)
		Node* parent = NULL;
		vector<Node*> children;
		board state;
		action::place parent_move;
		double UCT_val = 0;
		board::piece_type who;
	};

	void calculate_UCT(Node* n){
		//cout<<"x: "<<n->x<<endl;
		//cout<<"Tn: "<<n->Tn<<endl;
		//cout<<"n parent Tn: "<<n->parent->Tn<<endl;
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

		while(cur->children.size() != 0){
			double max_UCT = -1;
			Node* best_child = NULL;
			//cout<<"selection children size: "<<cur->children.size()<<endl;
			for(unsigned long int i=0;i<cur->children.size();i++){
				//cout<<"before UCT\n";
				calculate_UCT(cur->children[i]);
				//cout<<"after UCT\n";
				//cout<<"children UCT: "<<cur->children[i]->UCT_val<<endl;
				if(cur->children[i]->UCT_val > max_UCT){
					max_UCT = cur->children[i]->UCT_val;
					best_child = cur->children[i];
				}
				//cout<<"if end\n";
			}
			cur = best_child;
		}

		return cur;
	}

	void expansion(Node* n){
		board::piece_type nextWho = turn_who(n->who);
		//std::vector<action::place> cur_space = which_space(nextWho);
		std::shuffle(space.begin(), space.end(), engine);
		for (const action::place& move : space) {
			board after = n->state;
			if (move.color_apply(after, n->who) == board::legal){
				//cout<<"legal\n";
				Node* child = new Node;
				child->parent = n;
				n->children.push_back(child);
				child->state = after;
				child->who = nextWho;
				n->parent_move = move;
			}		
		}
		//cout<<"expand children size: "<<n->children.size()<<endl;
	}

	int simulation(Node* n){
		Node* cur = n;
		board cur_state = n->state;
		board::piece_type curWho = n->who;
		while(1){
			bool flag = false;
			//std::vector<action::place> cur_space = which_space(curWho);
			std::shuffle(space.begin(), space.end(), engine);
			for (const action::place& move : space) {
				board after = cur_state;
				if (move.color_apply(after, curWho) == board::legal){
					cur_state = after;
					curWho = turn_who(curWho);
					flag =true;
					break;
				}		
			}
			// terminal node
			if(!flag){
				if(cur->who == who){//lose
					return 0;
				}
				else{//win
					return 1;
				}
			}
		}
	}

	void backpropagation(Node* n, int re){
		Node* cur = n; //expansion node
		while(cur->parent != NULL){
			cur->Tn++;
			cur->x += re;
			cur = cur->parent;
		}
		//root
		cur->Tn++;
		cur->x += re;
	}

	void delete_tree(Node* n){
		if(n->children.size() == 0){
			delete n;
			return;
		}

		for(unsigned long int i=0;i<n->children.size();i++){
			delete_tree(n->children[i]);
		}
		n->children.clear();
		delete n;
		return;
	}

	void MCTS(Node* root){
		clock_t START_TIME, END_TIME;
		START_TIME = clock();
		int time = 0;

		while(1){
			Node* leaf = selection(root);
			//cout<<"finish selection\n";	
			expansion(leaf);
			//cout<<"finish expansion\n";
			Node* newNode;
			if(leaf->children.size() == 0){
				newNode = leaf;
			}
			else{
				std::random_device rd;
				std::default_random_engine rng(rd());
				shuffle(leaf->children.begin(), leaf->children.end(), rng);
				newNode = leaf->children[0];
			}
			int re = simulation(newNode);
			//cout<<"finish simulation\n";
			backpropagation(newNode, re);
			//cout<<"finish backpropagation\n";
			time++;

			// trminal condition
			if(timeout != -1){
				END_TIME = clock();
				if(timeout < END_TIME - START_TIME){
					break;
				}
				continue;
			}
			//cout<<"1\n";
			if(simulation_time != -1 && time == simulation_time){
				break;
			}
			//cout<<"2\n";
		}
	}

	virtual action take_action(const board& state) {
		//cout<<"***********************************agent check********************************************************************\n";
		Node* root = new Node;
		root->state = state;
		root->who = who;
		MCTS(root);
		//cout<<"MCTS FINISH\n";

		int max_count = -1;
		action best_move;
		int size = root->children.size();
		//cout<<"CHILDREN SIZE: "<<size<<endl;
		for(int i=0;i<size;i++){
			if(root->children[i]->Tn > max_count){
				max_count = root->children[i]->Tn;
				best_move = root->children[i]->parent_move;
			}
		}

		//root->children.clear();
		//delete root;
		delete_tree(root);

		return best_move;
	}

private:
	std::vector<action::place> space;// next legal move place
	std::vector<action::place> black_space;
	std::vector<action::place> white_space;
	board::piece_type who;
};

