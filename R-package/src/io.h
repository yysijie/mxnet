/*!
 *  Copyright (c) 2015 by Contributors
 * \file io.h
 * \brief Rcpp Data Loading and Iteration Interface of MXNet.
 */
#ifndef MXNET_RCPP_IO_H_
#define MXNET_RCPP_IO_H_

#include <Rcpp.h>
#include <mxnet/c_api.h>
#include <string>
#include "./base.h"

namespace mxnet {
namespace R {
// creator function of DataIter
class DataIterCreateFunction;

/*! \brief Base iterator interface */
class DataIter {
 public:
  /*! \return typename from R side. */
  inline static const char* TypeName() {
    return "DataIter";
  }
  /*! \brief Reset the iterator */
  virtual void Reset() = 0;
  /*!
   * \brief Move to next position.
   * \return whether the move is successful.
   */
  virtual bool Next() = 0;
  /*!
   * \brief number of padding examples.
   * \return number of padding examples.
   */
  virtual int NumPad() const = 0;
  /*!
   * \brief Get the Data Element
   * \return List of NDArray of elements in this value.
   */
  virtual Rcpp::List Value() const = 0;
  /*! \brief initialize the R cpp Module */
  static void InitRcppModule();
};

/*!
 * \brief MXNet's internal data iterator.
 */
class MXDataIter : public DataIter, MXNetMovable<MXDataIter> {
 public:
  /*! \return typename from R side. */
  inline static const char* TypeName() {
    return "MXNativeDataIter";
  }
  // implement the interface
  virtual void Reset();
  virtual bool Next();
  virtual int NumPad() const;
  virtual Rcpp::List Value() const;

 private:
  friend class DataIter;
  friend class DataIterCreateFunction;
  friend class MXNetMovable<MXDataIter>;
  // constructor
  MXDataIter() {}
  explicit MXDataIter(DataIterHandle handle)
      : handle_(handle) {}
  /*!
   * \brief create a R object that correspond to the Class
   * \param handle the Handle needed for output.
   */
  inline static Rcpp::RObject RObject(DataIterHandle handle) {
    return Rcpp::internal::make_new_object(new MXDataIter(handle));
  }
  // Create a new Object that is moved from current one
  inline MXDataIter* CreateMoveObject() {
    MXDataIter* moved = new MXDataIter();
    *moved = *this;
    return moved;
  }
  // finalizer that invoked on non-movable object
  inline void DoFinalize() {
    MX_CALL(MXDataIterFree(handle_));
  }
  /*! \brief internal data iter handle */
  DataIterHandle handle_;
};

/*!
 * \brief data iterator that takes a NumericVector
 *  Shuffles it and iterate over its content.
 *  TODO(KK, tq) implement this when have time.
 *  c.f. python/io.py:NDArrayIter
 */
class ArrayDataIter : public DataIter {
 public:
  /*! \return typename from R side. */
  inline static const char* TypeName() {
    return "MXArrayDataIter";
  }
  /*!
   * \brief Construct a ArrayDataIter from data and label.
   * \param data The data array.
   * \param label The label array.
   * \param batch_size The size of the batch.
   * \param shuffle Whether shuffle the data.
   */
  ArrayDataIter(const Rcpp::NumericVector& data,
                const Rcpp::NumericVector& label,
                int batch_size,
                bool shuffle);
  // implement the interface
  virtual void Reset() {}
  virtual bool Next() {
    return false;
  }
  virtual int NumPad() const {
    return 0;
  }
  virtual Rcpp::List Value() const {
    return Rcpp::List();
  }
};

/*! \brief The DataIterCreate functions to be invoked */
class DataIterCreateFunction : public ::Rcpp::CppFunction {
 public:
  virtual SEXP operator() (SEXP* args);

  virtual int nargs() {
    return 1;
  }

  virtual bool is_void() {
    return false;
  }

  virtual void signature(std::string& s, const char* name) {  // NOLINT(*)
    ::Rcpp::signature< SEXP, ::Rcpp::List >(s, name);
  }

  virtual const char* get_name() {
    return name_.c_str();
  }

  virtual SEXP get_formals() {
    return Rcpp::List::create(Rcpp::_["alist"]);
  }

  virtual DL_FUNC get_function_ptr() {
    return (DL_FUNC)NULL; // NOLINT(*)
  }
  /*! \brief static function to initialize the Rcpp functions */
  static void InitRcppModule();

 private:
  // make constructor private
  explicit DataIterCreateFunction(DataIterCreator handle);
  /*! \brief internal creator handle. */
  DataIterCreator handle_;
  // name of the function
  std::string name_;
};
}  // namespace R
}  // namespace mxnet

RCPP_EXPOSED_CLASS_NODECL(::mxnet::R::MXDataIter);
RCPP_EXPOSED_CLASS_NODECL(::mxnet::R::ArrayDataIter);

namespace Rcpp {
  template<>
  inline bool is<mxnet::R::MXDataIter>(SEXP x) {
    return internal::is__module__object_fix<mxnet::R::MXDataIter>(x);
  }
  template<>
  inline bool is<mxnet::R::ArrayDataIter>(SEXP x) {
    return internal::is__module__object_fix<mxnet::R::ArrayDataIter>(x);
  }
  // This patch need to be kept even after the Rcpp update merged in.
  template<>
  inline bool is<mxnet::R::DataIter>(SEXP x) {
    return is<mxnet::R::MXDataIter>(x) ||
        is<mxnet::R::ArrayDataIter>(x);
  }
}  // namespace Rcpp
#endif  // MXNET_RCPP_IO_H_

