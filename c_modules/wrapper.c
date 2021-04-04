#include <Python.h>
#include <stdbool.h>

#include "utils.h"

static PyObject* str2bool_wrapper(PyObject* self, PyObject* args) {
    char* input;

    // parse arguments
    if (!PyArg_ParseTuple(args, "s", &input))
        return NULL;
    
    int8_t len = strlen(input);
    int8_t chars[len];
    for(int8_t i = 0; i < len; i++) chars[i] = (int8_t) input[i];
    //! Not able to compare proparly. Potential encoding difference.
    // build the resulting bool into a Python object and return
    return Py_BuildValue("O", str2bool(chars, len) ? Py_True : Py_False);
    //return Py_BuildValue("N", PyBool_FromLong(str2bool(input)));
    //return Py_BuildValue("s", input);
    //return Py_BuildValue("i", str2bool(input));
}

static PyObject* version(PyObject*self) {
    return Py_BuildValue("s", "Version 0.2-dev");
}

/**
 * Defines how actually function will be called.
*/
static PyMethodDef Methods[] = {
    { "str2bool", str2bool_wrapper, METH_VARARGS,
        "Returns a boolean from string" },
    // ...
    { "version", (PyCFunction)version, METH_NOARGS,
        "Prints C lib versioning"},
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef Utils = {
    PyModuleDef_HEAD_INIT,
    "utils",
    "Util Module",
    -1,
    Methods

};

PyMODINIT_FUNC PyInit_utils(void) {
    return PyModule_Create(&Utils);
}
// DL_EXPORT(void) init(void) {    
//     Py_InitModule("str2bool", Methods);
// }