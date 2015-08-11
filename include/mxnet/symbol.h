/*!
 *  Copyright (c) 2015 by Contributors
 * \file symbol.h
 * \brief symbolic interface of mxnet
 */
#ifndef MXNET_SYMBOL_H_
#define MXNET_SYMBOL_H_

#include <mxnet/atomic_symbol.h>
#include <algorithm>
#include <vector>
#include <memory>
#include <queue>
#include <string>
#include <iostream>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include "./base.h"
#include "./tensor_blob.h"
#include "./operator.h"
#include "./static_graph.h"

namespace mxnet {
/*!
 * \brief Symbol is used to represent dynamically generated symbolic computation graph.
 *
 *   This class is used as a tool to generate computation graphs(aka. configuration) of the network.
 *   Symbol is always composite, the head Node is the output node of the symbol.
 *   An atomic symbol can be seen as a special case of the composite symbol with only the head node.
 *
 *   The symbol can be converted from/to StaticGraph, the actual configuration used by mxnet.
 *   Symbol offers more flexible way to composite nodes than StaticGraph, which makes it good
 *   tool to generate configurations from language bindings such as python.
 * \sa StaticGraph
 */
class Symbol {
 public:
  /*!
   * \brief copy the symbol
   * \return a deep copy of the graph
   */
  Symbol Copy() const;
  /*!
   * \brief print the symbol info to output stream.
   * \param os the output stream we like to print to
   */
  void Print(std::ostream &os) const; // NOLINT(*)
  /*!
   * \brief List the arguments names.
   *
   * The position of the returned list also corresponds to calling position in operator()
   * \return the arguments list of this symbol, they can be either named or unnamed (empty string).
   */
  std::vector<std::string> ListArguments() const;
  /*! \return get the descriptions of outputs for this symbol */
  std::vector<std::string> ListReturns() const;
  /*!
   * \brief get the index th element from the returned tuple.
   * \param index index of multi output
   * \return the symbol corresponds to the indexed element.
   */
  Symbol operator[] (int index) const;
  /*!
   * \brief Compose the symbol with arguments, this changes current symbol.
   *
   * The positional arguments passed in must be complete(contain all arguments).
   *
   * \param args positional arguments for the symbol
   * \param name name of returned symbol.
   */
  void Compose(const std::vector<Symbol>& args,
               const std::string& name);
  /*!
   * \brief Compose the symbol with arguments, this changes the current symbol.
   * The kwargs passed in can be in-complete,
   *
   * The rest of the symbols will remain the same name.
   *
   * \param kwargs keyword arguments for the symbol
   * \param name name of returned symbol.
   */
  void Compose(const std::unordered_map<std::string, Symbol>& kwargs,
               const std::string& name);
  /*!
   * \brief Apply the symbol as a function, compose with arguments
   * \param args positional arguments for the symbol
   * \param name name of returned symbol.
   * \return a new Symbol which is the composition of current symbol with its arguments
   */
  inline Symbol operator () (const std::vector<Symbol>& args,
                             const std::string& name) const {
    Symbol s = this->Copy();
    s.Compose(args, name);
    return s;
  }
  /*!
   * \brief compose with named arguments
   * \param kwargs keyword arguments for the symbol
   * \param name name of returned symbol.
   * \return a new symbol which is the composition of current symbol with its arguments
   */
  inline Symbol operator () (const std::unordered_map<std::string, Symbol>& kwargs,
                             const std::string& name) {
    Symbol s = this->Copy();
    s.Compose(kwargs, name);
    return s;
  }
  /*!
   * \brief infer the shapes of outputs and unknown input arguments
   * \param in_shape the shape of input arguments of the operator
   *     this should be of same length as the vector returned by ListArguments
   *     in_shape allows unknown elements, which are checked by shape.ndim() == 0.
   *     For unknown shapes, InferShape will try to fill in the correct Shape in in_shape
   *     For known shapes, InferShape will check shape consistency
   *
   *     common practice: set the shape of data input, and usually weight's shape can be infered
   *
   * \param out_shape the shape of outputs of the operator
   *     InferShape will modify the vector to fill output TShape
   * \return if the shape inference is successful, return true, else return false.
   */
  inline bool InferShape(std::vector<TShape> *in_shape,
                         std::vector<TShape> *out_shape) {
    StaticGraph g;
    Symbol::Convert({*this}, &g);
    return g.InferShape(in_shape, out_shape);
  }
  /*!
   * \brief create Symbol by wrapping AtomicSymbol
   * This function takes the ownership of atomic_symbol.
   *
   * \param atomic_symbol the AtomicSymbol
   * \return Symbol
   * \sa AtomicSymbol::Create
   */
  static Symbol Create(AtomicSymbol *atomic_symbol);
  /*!
   * \brief create equivalence of symbols from static graphs
   * \param graph the static graph
   * \return list of Symbols representing outputs of the graph
   */
  static std::vector<Symbol> Create(const StaticGraph &graph);
  /*!
   * \brief Convert a list of symbols into static graph
   *
   *  The user can go further to call bind function on static graph
   *
   * \param heads the heads of the graph
   * \param out_graph the pointer holder of the output graph
   */
  static void Convert(const std::vector<Symbol> &heads, StaticGraph *out_graph);
  /*!
   * \brief create variable symbol node
   * \param name name of the variable
   * \return the new variable
   */
  inline static Symbol CreateVariable(const std::string &name) {
    Symbol s;
    s.head_ = DataEntry(std::make_shared<Node>(nullptr, name), 0);
    return std::move(s);
  }

 protected:
  // forward declare Node
  struct Node;
  /*! \brief an entry that represents output data from a node */
  struct DataEntry {
    /*! \brief the source node of this data */
    std::shared_ptr<Node> source;
    /*!
     * \brief index of output from the source.
     * If index == -1, it represents all the outputs.
     */
    int index;
    /*! \brief enabled default copy constructor */
    DataEntry() {}
    /*! \brief constructor from index */
    DataEntry(std::shared_ptr<Node> source, int index)
        : source(source), index(index) {}
  };
  /*!
   * \brief Node is represents node of an operator in the symbolic graph.
   *
   * It stores connection to the inputs to function represented by AtomicSymbol
   * NOTE on data structure: there are three types of node:
   * - Normal node: contains all the necessary elements of a graph.
   * - AtomicSymbol: the inputs_ is empty, represents an AtomicSymbol that has not been applied.
   * - Variable: the sym_ is nullptr, represents an named Variable of tensors that can be composed.
   */
  struct Node {
    /*! \brief wrapped atomic symbol */
    std::unique_ptr<AtomicSymbol> sym;
    /*! \brief name of the node */
    std::string name;
    /*! \brief inputs to this node */
    std::vector<DataEntry> inputs;
    /*!
     * \brief constructor
     * \param sym the AtomicSymbol to construct the symbol
     * \param name the name of the symbol
     */
    explicit Node(AtomicSymbol* sym = nullptr, const std::string& name = "")
        : sym(sym), name(name) {
    }
    /*! \return Whether the symbol is AtomicSymbol */
    inline bool is_atomic() const {
      return inputs.size() == 0 && sym != nullptr;
    }
    /*! \return Whetehr the symbolc is a PlaceHolder */
    inline bool is_variable() const {
      return sym == nullptr;
    }
  };
  /*! \brief the head node of the Symbol */
  DataEntry head_;

 private:
  /*! \brief DFS Visit for symbol with single head
   *   This function call is specail case for DFSVisit_
   *  \param fvisit function applied for each visit.
   *  \tparam FVisit visiting function type
   */
  template<typename FVisit>
  inline void DFSVisit(FVisit fvisit) const {
    DFSVisit({*this}, fvisit);
  }
  /*!
   * \brief Visit all the nodes in left-to-right depth first order.
   *
   *  This function will visit the graph in DFS order, call fvisit exactly once
   *  for each Node, and store the result in out_result.
   *
   * \param fvisit function applied for each visit.
   * \tparam FVisit visiting function type
   */
  template<typename FVisit>
  static inline void DFSVisit(const std::vector<Symbol> &heads,
                              FVisit fvisit) {
    std::vector<Node*> stack;
    std::unordered_set<Node*> visited;
    // put the head into the graph
    for (auto &head : heads) {
      Node *ptr = head.head_.source.get();
      if (visited.count(ptr) == 0) {
        stack.push_back(ptr);
        visited.insert(ptr);
      }
    }
    while (!stack.empty()) {
      Node* back = stack.back();
      stack.pop_back();
      fvisit(back);
      for (auto it = back->inputs.rbegin(); it != back->inputs.rend(); ++it) {
        Node *ptr = it->source.get();
        if (visited.count(ptr) == 0) {
          stack.push_back(ptr);
          visited.insert(ptr);
        }
      }
    }
  }
  /*!
   * \brief Find duplicate arguments in the composition
   * \param out the map of argument-name -> occurence count
   * \return maximum number of duplication factor
   */
  int FindDuplicateArgs(std::unordered_map<std::string, int> *out) const;
};
}  // namespace mxnet
#endif  // MXNET_SYMBOL_H_