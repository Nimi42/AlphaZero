from setuptools import setup, Extension

module1 = Extension('cmcts',
                    sources=[
                        'src/test.c'],
                    # include_dirs=[
                    #     'mcts/include',
                    #     'mcts/str_node_dict/include',
                    #     'mcts/multiline/include'
                    # ],
                    extra_compile_args=['-D DEBUG', '-D NULL_CHECKS', '-fms-extensions'])
                    #extra_objects=['/home/nemo/PycharmProjects/mcts/mcts/str-node-dict.o'])

setup(name='PackageName',
      version='1.0',
      description='This is a demo package',
      ext_modules=[module1])
