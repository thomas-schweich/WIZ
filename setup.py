from distutils.core import setup
import py2exe
import matplotlib

setup(
    windows=[
        {'script': 'WIZ.py',
         'icon_resources': [(1, r'C:\Users\thoma\PycharmProjects\dataManipulation\res\WIZ.ico')]
         }
    ],
    data_files=matplotlib.get_py2exe_datafiles(),
    zipfile=None,
    options={
        'py2exe': {
            'bundle_files': 3,
            'includes': ['Tkinter',
                         'scipy', 'scipy.special.*', 'scipy.linalg.*', 'scipy.integrate',
                         'scipy.sparse.csgraph._validation'],
            'excludes': ["matplotlib.backends.backend_qt4agg"],
        }
    }
)
