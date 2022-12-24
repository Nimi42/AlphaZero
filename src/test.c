#include <Python.h>
#include <structmember.h>


// ##########################                   C implementation

typedef struct Node {
    PyObject *state;

    double P_sa;
    int Q_sa, N_sa;

    struct Node* children;
    size_t num_children;
} Node;

typedef struct NodeResult {
    Node;
    PyObject* action;
} NodeResult;

static void new_node(PyObject* state, Node** node) {
    // Create the node
    Node* tmp = malloc(sizeof(*node));
    tmp->state=state;
    tmp->P_sa = 0;
    tmp->Q_sa = 0;
    tmp->N_sa = 0;
    tmp->num_children = 0;
    *node = tmp;
}

//typedef struct SearchPath {
//    union {
//        struct Node *node;
//        struct NodeResult res;
//    };
//    struct SearchPath* next;
//} SearchPath;

//static void execute_round(Node node);
//
//static Node _select(Node parent, (*score_func)(Node Parent, Node Child));

// ##########################                   Class Definition

typedef struct {
    PyObject_HEAD
    PyObject* module;
    int iterations;
} cmctsObject;


static PyObject * cmcts_search(PyObject *self, PyObject *state) {
    NodeResult *root;
    new_node(state, (Node**) &root);
    printf("P_sa: %f \nQ_sa: %d \nN_sa: %d \nNum_Children: %zu\n\n", root->P_sa, root->Q_sa, root->N_sa, root->num_children);

    Node* a = (Node*) root;
    printf("P_sa: %f \nQ_sa: %d \nN_sa: %d \nNum_Children: %zu\n\n", a->P_sa, a->Q_sa, a->N_sa, a->num_children);
    root = (Node*) a;
    printf("P_sa: %f \nQ_sa: %d \nN_sa: %d \nNum_Children: %zu\n\n", root->P_sa, root->Q_sa, root->N_sa, root->num_children);




//    PyObject* from_list = PyList_New(1);
//    PyObject* python_str = Py_BuildValue("s", "log");
//    PyList_SetItem(from_list, 0, python_str);

//    PyObject* objectsRepresentation = PyObject_Repr(from_list);
//    const char* s = PyUnicode_AsUTF8(objectsRepresentation);
//    printf(s);

//    PyObject* math =  PyImport_ImportModule("math");
//    PyObject* python_str = Py_BuildValue("s", "log");
//
//    PyObject* log = PyObject_GetAttr(math, python_str);
//    PyObject* ten = Py_BuildValue("(i)", 10);
//
//    PyObject* res = PyObject_CallObject(log, ten);
//
//    PyObject* objectsRepresentation = PyObject_Repr(res);
//    const char* s = PyUnicode_AsUTF8(objectsRepresentation);
//    printf(s);
////    PyObject* abc = PyImport_ImportModuleLevelObject("log", NULL, NULL, "math");


    return Py_None;
}

static PyMethodDef cmctsObj_methods[] = {
    { "search", (PyCFunction) cmcts_search, METH_O, "search method" },
    { NULL }
};

static PyTypeObject cmctsType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "cmcts.Cmcts",             /* tp_name */
    sizeof(cmctsObject), /* tp_basicsize */
    0,                         /* tp_itemsize */
    0,                         /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "cmcts objects",           /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    cmctsObj_methods,                         /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    PyType_GenericNew,         /* tp_new */
};



// ##########################                   Module Definitiontmp

static PyModuleDef cmctsmodule = {
    PyModuleDef_HEAD_INIT,
    "cmcts",
    "Example module that creates an extension type.",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit_cmcts(void) {
    PyObject* m;

    if (PyType_Ready(&cmctsType) < 0)
        return NULL;

    m = PyModule_Create(&cmctsmodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&cmctsType);
    PyModule_AddObject(m, "Cmcts", (PyObject *)&cmctsType);
    return m;
}
