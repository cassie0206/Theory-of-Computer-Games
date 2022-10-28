#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include "board.h"
#include "action.h"
#include "weight.h"
#include <fstream>

int operation;
std::vector<board::cell> bag;

const int tuple_count = 4;
int t_element_count[tuple_count] = {6, 6, 4, 4};
int tuple[4][6] = { {0, 4, 8, 1, 5, 9},
                    {1, 5, 9, 2, 6, 10},
                    {2, 6, 10, 14},
                    {3, 7, 11, 15}};

/*
const int tuple_count = 8;
const int t_element_count = 4;
int tuple[8][4] = { {0, 1, 2, 3},
                    {4, 5, 6, 7},
                    {8, 9, 10, 11},
                    {12, 13, 14, 15},
                    {0, 4, 8, 12},
                    {1, 5, 9, 13},
                    {2, 6, 10, 14},
                    {3, 7, 11, 15}};
*/
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
 * base agent for agents with weight tables
 */
class weight_agent : public agent {
public:
    weight_agent(const std::string& args = "") : agent(args) {
        if (meta.find("init") != meta.end()) // pass init=... to initialize the weight
            init_weights(meta["init"]);
        if (meta.find("load") != meta.end()) // pass load=... to load from a specific file
            load_weights(meta["load"]);
    }
    virtual ~weight_agent() {
        if (meta.find("save") != meta.end()) // pass save=... to save to a specific file
            save_weights(meta["save"]);
    }

protected:
    virtual void init_weights(const std::string& info) {
        net.emplace_back(65536); // create an empty weight table with size 65536
        net.emplace_back(65536); // create an empty weight table with size 65536
        net.emplace_back(65536); // create an empty weight table with size 65536
        net.emplace_back(65536); // create an empty weight table with size 65536
        net.emplace_back(65536); // create an empty weight table with size 65536
        net.emplace_back(65536); // create an empty weight table with size 65536
        net.emplace_back(65536); // create an empty weight table with size 65536
        net.emplace_back(65536); // create an empty weight table with size 65536
        // now net.size() == 2; net[0].size() == 65536; net[1].size() == 65536
    }
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
};

/**
 * base agent for agents with a learning rate
 */
class learning_agent : public agent {
public:
    learning_agent(const std::string& args = "") : agent(args), alpha(0.1f) {
        if (meta.find("alpha") != meta.end())
            alpha = float(meta["alpha"]);
    }
    virtual ~learning_agent() {}

protected:
    float alpha;
};

/**
 * random environment
 * add a new random tile to an empty cell
 * 2-tile: 90%
 * 4-tile: 10%
 */
class rndenv : public random_agent {
public:
    rndenv(const std::string& args = "") : random_agent("name=random role=environment " + args),
        space({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }), popup(0, 9) {}

    virtual action take_action(const board& after) {
        if (bag.empty()) {
            for (int i = 1; i <= 3; i++)
                bag.push_back(i);
            std::random_shuffle(bag.begin(), bag.end());
        }
        board::cell tile = bag.back();
        bag.pop_back();

        std::vector<int> legalspace;
        switch (operation) {
            case 0: legalspace = {12, 13, 14, 15}; break;
            case 1: legalspace = {0, 4, 8, 12}; break;
            case 2: legalspace = {0, 1, 2, 3}; break;
            case 3: legalspace = {3, 7, 11, 15}; break;
            default: legalspace = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        }

        std::shuffle(legalspace.begin(), legalspace.end(), engine);
        
        for (int pos : legalspace) {
            if (after(pos) != 0) continue;
            return action::place(pos, tile);
        }
        return action();
    }

private:
    std::array<int, 16> space;
    std::uniform_int_distribution<int> popup;
};

/**
 * dummy player
 * select a legal action randomly
 */
class player : public weight_agent {
public:
    player(const std::string& args = "") : weight_agent("name=dummy role=player " + args),
        opcode({ 0, 1, 2, 3 }) {
            for (int i=0; i<8; i++) {
                net.emplace_back(weight(15*15*15*15*15*15*15));
            }
        }

    unsigned encode(const board& state, int i) const {
        int* t = tuple[i];
        switch (i) {
            case 0: case 1: // the larger part (4 * 2)
                return (state(t[0]) << 0) | (state(t[1]) << 4) | (state(t[2]) << 8) | (state(t[3]) << 12) | (state(t[4]) << 16) | (state(t[5]) << 20);
            default: // the smaller part (4 * 1)
                return (state(t[0]) << 0) | (state(t[1]) << 4) | (state(t[2]) << 8) | (state(t[3]) << 12);
        }
    }

    void reflect_tuple(int i) const {
        int reflect[16] = { 3, 2, 1, 0,
                            7, 6, 5, 4, 
                            11, 10, 9, 8, 
                            15, 14, 13, 12};
        for (int j = 0; j < t_element_count[i]; j++) 
            tuple[i][j] = reflect[ tuple[i][j] ];
    }

    void rotate_tuple(int i) const {
        int rotate[16] = {  3, 7, 11, 15,
                            2, 6, 10, 14, 
                            1, 5, 9, 13, 
                            0, 4, 8, 12};                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        for (int j = 0; j < t_element_count[i]; j++)
            tuple[i][j] = rotate[ tuple[i][j] ];
    }

    float get_board_value(const board& state) const {
        float v = 0;
        for (int i = 0; i < tuple_count; i++) {
            for (int rf = 0; rf < 2; rf++) {
                reflect_tuple(i);
                for (int rt = 0; rt < 4; rt++) {
                    rotate_tuple(i);
                    v += net[i][encode(state, i)];
                }
            }
        }
        return v;
    }

    void train_weight(board::reward reward) {
        double alpha = 0.003125;
        double v_s = alpha * (get_board_value(next) - get_board_value(previous) + reward);
        if (reward == -1) v_s = alpha * (-get_board_value(previous));
        for (int i = 0; i < tuple_count; i++) {
            for (int rf = 0; rf < 2; rf++) {
                reflect_tuple(i);
                for (int rt = 0; rt < 4; rt++) {
                    rotate_tuple(i);
                    net[i][encode(previous, i)] += v_s;
                }
            }
        }
    }

    virtual void open_episode(const std::string& flag = "") {
        count = 0;
    }

    virtual action take_action(const board& before) {
        float bestvalue = -999999999;
        int bestop = -1;
        for (int op = 0; op < 4; op++) {
            board temp = before;
            board::reward reward = temp.slide(op);
            float value = get_board_value(temp);
            if (bestop == -1 && reward != -1)
                bestop = op;
            if (reward + value > bestvalue && reward != -1) {
                bestvalue = reward + value;
                bestop = op;
            }
        }
        if (bestop != -1) {
            next = before;
            board::reward reward = next.slide(bestop);
            if (count) train_weight(reward);
            previous = next;
            count++;
            return action::slide(operation = bestop);
        } else {
            train_weight(-1);
            return action();
        }
    }

private:
    std::array<int, 4> opcode;
    board previous;
    board next;
    int count;
};