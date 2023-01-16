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
#include <climits>
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
			timeout = (int(meta["timeout"]));
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
		//for (size_t i = 0; i < space.size(); i++)
		//	space[i] = action::place(i, who);
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
		double UCT_val = LLONG_MAX;
		board::piece_type who;
	};

	void calculate_UCT(Node* n){
		if(n->x == 0 || n->Tn == 0) return;
		n->UCT_val = (double)((double)n->x / n->Tn) + 0.5 * (double)sqrt( (double)log((double)n->parent->Tn) / n->Tn);
	}

	Node* selection(Node* root){
		Node* cur = root;

		while(cur->children.size() != 0){
			double max_UCT = -1;
			Node* best_child = NULL;
			
			for(size_t i=0;i<cur->children.size();i++){
				if (cur->children[i]->Tn == 0){
					// not explore
					best_child = cur->children[i];
					return best_child;
				}

				calculate_UCT(cur->children[i]);
				if(cur->children[i]->UCT_val > max_UCT){
					max_UCT = cur->children[i]->UCT_val;
					best_child = cur->children[i];
				}
			}
			cur = best_child;
		}
		return cur;
	}
		
	void expansion(Node* n) {
		board::piece_type nextWho = (n->who == board::black ? board::white : board::black);
	 	
		if (nextWho == board::white) {
			for(size_t i=0;i<white_space.size();i++){
				board after = n->state;
				if(white_space[i].apply(after) != board::legal){
					continue;
				}
				Node* child = new Node;
				child->state = after;
				child->parent = n;
				child->parent_move = white_space[i];
				child->who = nextWho;
				n->children.push_back(child);
			}
		}
		else if (nextWho == board::black) {
			for(size_t i=0;i<black_space.size();i++){
				board after = n->state;
				if(black_space[i].apply(after) != board::legal){
					continue;
				}
				Node* child = new Node;
				child->state = after;
				child->parent = n;
				child->parent_move = black_space[i];
				child->who = nextWho;
				n->children.push_back(child);
			}
		}
	}

	int simulation(Node* n){
		board::piece_type curWho = n->who;
		board cur_state = n->state;

		while(1){
			curWho = (curWho == board::black ? board::white : board::black);
			if(curWho == board::black){
				std::shuffle(black_space.begin(), black_space.end(), engine);
				for(unsigned long int i=0;i<black_space.size();i++){// continue playing without recording
					board after = cur_state;
					if(black_space[i].apply(after) == board::legal){
						cur_state = after;
						break;
					}
					if(i == black_space.size() - 1){
						// if the same with current player(who), loss. Otherwise, win
						// since if the same means that current player can't continue to play
						return who == board::white ? 1 : 0;
					}
				}		
			}
			else{
				std::shuffle(white_space.begin(), white_space.end(), engine);
				for(unsigned long int i=0;i<white_space.size();i++){// continue playing without recording
					board after = cur_state;
					if(white_space[i].apply(after) == board::legal){
						cur_state = after;
						break;
					}
					if(i == white_space.size() - 1){
						// if the same with current player(who), loss. Otherwise, win
						// since if the same means that current player can't continue to play
						return who == board::black ? 1 : 0;
					}
				}	
			}

		}
	}

	void backpropagation(Node* root, Node* cur, int re) {
		root->x+=re;
		root->Tn++;
		while(cur != NULL && cur != root) {
			cur->Tn++;
			cur->x+=re;
			cur = cur->parent;
		}
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
		int time = 0;
		
		while(1){
			Node* leaf = selection(root);
			expansion(leaf);
			Node* newNode;
			if(leaf->children.size() == 0){
				newNode = leaf;
			}
			else{
				std::shuffle(leaf->children.begin(), leaf->children.end(), engine);
				newNode = leaf->children[0];
			}
			int re = simulation(newNode);
			backpropagation(root, newNode, re);
			time++;

			if(simulation_time != -1 && time == simulation_time){
				break;
			}
		}
	}

	virtual action take_action(const board& state) {
		Node* root = new Node;
		root->state = state;
		root->who = (who == board::white ? board::black : board::white);
		MCTS(root);

		int max_count = -1;
		action best_move;
		int size = root->children.size();
		for(int i=0;i<size;i++){
			if(root->children[i]->Tn > max_count){
				max_count = root->children[i]->Tn;
				best_move = root->children[i]->parent_move;
			}
		}

		delete_tree(root);
		
		return best_move;
	}

private:
	std::vector<action::place> space;// next legal move place
	std::vector<action::black> black_space;
	std::vector<action::white> white_space;
	board::piece_type who;
};