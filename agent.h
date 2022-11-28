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

struct v{
	int total = 0;
	int win = 0;
};
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
		double UCT_val = 0x7fffffff;
		board::piece_type who;
	};

	void calculate_UCT(Node* n, int total_visit_count){
		// cout<<"x: "<<n->x<<endl;
		// cout<<"Tn: "<<n->Tn<<endl;
		//cout<<"n parent Tn: "<<n->parent->Tn<<endl;
		if(n->x == 0 || n->Tn == 0) return;
		n->UCT_val = (double)((double)n->x / n->Tn) + 0.5 * (double)sqrt( (double)log((double)total_visit_count) / n->Tn);
	}

	board::piece_type turn_who(board::piece_type who){
		if(who == board::black){
			return board::white;
		}
		else{
			return board::black;
		}
	}

	/*std::vector<action::place> which_space(board::piece_type who){
		if(who == board::black){
			return black_space;
		}
		else{
			return white_space;
		}
	}*/

	Node* selection(Node* root, int total_visit_count){
		Node* cur = root;
		int idx = -1;
		while(cur->children.size() != 0){
			double max_UCT = -1;
			Node* best_child = NULL;
			// cout<<"selection children size: "<<cur->children.size()<<endl;
			
			for(size_t i=0;i<cur->children.size();i++){
				//cout<<"before UCT\n";
				// calculate_UCT(cur->children[i], cur->Tn);
				//cout<<"after UCT\n";
				//cout<<"children UCT: "<<cur->children[i]->UCT_val<<endl;
				if (cur->children[i]->Tn == 0){
					idx = i;
					best_child = cur->children[i];
					return best_child;
					// break;
				}
				if(cur->children[i]->UCT_val > max_UCT){
					// cout << i << endl;
					max_UCT = cur->children[i]->UCT_val;
					best_child = cur->children[i];
				}
				//cout<<"if end\n";
			}
			// cout << max_UCT << endl;
			cur = best_child;
		}
		// cout << idx <
		return cur;
	}

	// void expansion(Node* n){// n is selected leaf node
	// 	//board::piece_type nextWho = (n->who == board::black ? board::white : board::black);
	// 	//std::vector<action::place> cur_space = which_space(nextWho);
	// 	//std::shuffle(space.begin(), space.end(), engine);		
	// 	/*for (const action::place& move : space) {
	// 		board after = n->state;
	// 		if (move.color_apply(after, n->who) == board::legal){
	// 			//cout<<"legal\n";
	// 			Node* child = new Node;
	// 			child->parent = n;
	// 			child->state = after;
	// 			child->who = nextWho;
	// 			child->parent_move = move;
	// 			n->children.push_back(child);
	// 		}		
	// 	}*/
	// 	/*for(unsigned long int i=0;i<space.size();i++){
	// 		board after = n->state;
	// 		if(space[i].color_apply(after, n->who) != board::legal){
	// 			continue;
	// 		}
	// 		Node* child = new Node;
	// 		child->state = after;
	// 		child->parent = n;
	// 		child->parent_move = space[i];
	// 		child->who = nextWho;
	// 		n->children.push_back(child);
	// 	}*/
	// 	/*std::vector<action::place> cur_space = (n->who == board::black ? black_space : white_space);
	// 	for(unsigned long int i=0;i<cur_space.size();i++){
	// 		board after = n->state;
	// 		if(cur_space[i].apply(after) != board::legal){
	// 			continue;
	// 		}
	// 		Node* child = new Node;
	// 		child->state = after;
	// 		child->parent = n;
	// 		child->parent_move = cur_space[i];
	// 		child->who = nextWho;
	// 		n->children.push_back(child);
	// 	}*/
	// 	//cout<<"expand children size: "<<n->children.size()<<endl;
	// 	board::piece_type nextWho = (n->who == board::black ? board::white : board::black);
	// 	if(nextWho == board::black){
	// 		for(size_t i=0;i<white_space.size();i++){
	// 			board after = n->state;
	// 			if(white_space[i].apply(after) != board::legal){
	// 				continue;
	// 			}
	// 			Node* child = new Node;
	// 			child->state = after;
	// 			child->parent = n;
	// 			child->parent_move = white_space[i];
	// 			child->who = nextWho;
	// 			n->children.push_back(child);
	// 		}
	// 	}
	// 	else{
	// 		for(size_t i=0;i<black_space.size();i++){
	// 			board after = n->state;
	// 			if(black_space[i].apply(after) != board::legal){
	// 				continue;
	// 			}
	// 			Node* child = new Node;
	// 			child->state = after;
	// 			child->parent = n;
	// 			child->parent_move = black_space[i];
	// 			child->who = nextWho;
	// 			n->children.push_back(child);
	// 		}
	// 	}
	// }
		
	void expansion(Node* parent_node) {
		board::piece_type child_who;
		action::place child_move;
	 	
		if (parent_node->who == board::black) {
			child_who = board::white;
			for(const action::place& child_move : white_space) {
				board after = parent_node->state;
				if (child_move.apply(after) == board::legal) {
					Node* child_node = new Node;
					child_node->state = after;
					child_node->parent = parent_node;
					child_node->parent_move = child_move;
					child_node->who = child_who;
					
					parent_node->children.emplace_back(child_node);
				}
			}
		}
		else if (parent_node->who == board::white) {
			child_who = board::black;
			for(const action::place& child_move : black_space) {
				board after = parent_node->state;
				if (child_move.apply(after) == board::legal) {
					Node* child_node = new Node;
					child_node->state = after;
					child_node->parent = parent_node;
					child_node->parent_move = child_move;
					child_node->who = child_who;
					
					parent_node->children.emplace_back(child_node);
				}
			}
		}
			
	}

	board::piece_type simulation(Node* n){
		/*board cur_state = n->state;
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
				if(curWho == who){//lose
					return 1;
				}
				else{//win
					return 0;
				}
			}
		}*/
		board::piece_type curWho = n->who;
		board cur_state = n->state;

		while(1){
			//std::vector<action::place> cur_space = (curWho == board::black ? black_space : white_space);
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
						return board::white;
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
						return board::black;
					}
				}	
			}

		}
	}

	// void backpropagation(Node* root, Node* n, int re){
	// 	Node* cur = n; //expansion node
	// 	while(cur != NULL && cur != root){
	// 		cur->Tn++;
	// 		cur->x += re;
	// 		cur = cur->parent;
	// 	}
	// 	root->Tn++;
	// 	root->x += re;
	// }
	void backpropagation(Node* root, Node* cur, board::piece_type winner) {
		// root state : last_action = white 
		// -> root who = black 
		bool win = true;
		if(winner == root->who){
			win = false;
			root->x++;
		}
		root->Tn++;
		while(cur != NULL && cur != root) {
			cur->Tn++;
			if(win == true){
				cur->x++;
			}
			calculate_UCT(cur, cur->parent->Tn+1);
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
		clock_t START_TIME, END_TIME;
		START_TIME = clock();
		int time = 0;
		
		int total_visit_count = 0;
		// cout << timeout << endl;
		expansion(root);
		
		int cnt = 0;		
		while(1){
			Node* leaf = selection(root, total_visit_count);
			total_visit_count++;
			//cout<<"finish selection\n";	
			expansion(leaf);
			// if(time==10) cout << leaf->parent->children.size() << endl;
			//cout<<"finish expansion\n";
			Node* newNode;
			if(leaf->children.size() == 0){
				newNode = leaf;
			}
			else{
				std::shuffle(leaf->children.begin(), leaf->children.end(), engine);
				newNode = leaf->children[0];
			}
			board::piece_type re = simulation(newNode);
			//cout<<"finish simulation\n";
			backpropagation(root, newNode, re);
			//cout<<"finish backpropagation\n";
			time++;

			// trminal condition
			// if(timeout != -1){
			// 	END_TIME = clock();
			// 	if(timeout < END_TIME - START_TIME){
			// 		break;
			// 	}
			// 	continue;
			// }
			//cout<<"1\n";
			if(simulation_time != -1 && time == simulation_time){
				break;
			}
			//cout<<"2\n";
		}
	}

	virtual action take_action(const board& state) {
		// cout<<"***********************************agent check********************************************************************\n";
		Node* root = new Node;
		root->state = state;
		// root->who = who;
		root->who = (who == board::white ? board::black : board::white);
		MCTS(root);
		// cout<<"MCTS FINISH\n";

		int max_count = -1;
		action best_move;
		int size = root->children.size();
		//double max_UCT = -1;
		// cout<<"CHILDREN SIZE: "<<size<<endl;
		for(int i=0;i<size;i++){
			if(root->children[i]->Tn > max_count){
				max_count = root->children[i]->Tn;
				best_move = root->children[i]->parent_move;
			}
		}
		// if(size == 0){
		// 	return action();
		// }
		//root->children.clear();
		//delete root;
		delete_tree(root);
		// free(root);
		return best_move;
	}

private:
	std::vector<action::place> space;// next legal move place
	std::vector<action::place> black_space;
	std::vector<action::place> white_space;
	board::piece_type who;
	std::map<action::place, v> action2v;
};