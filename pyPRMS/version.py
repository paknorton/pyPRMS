major = 0
minor = 9
micro = 7
__version__ = f"{major}.{minor}.{micro}"

__pakname__ = "pyPRMS"

# edit author dictionary as necessary (in order of commits after Bakker and Post)
author_dict = {
    "Parker Norton": "pnorton@usgs.gov",
}
__author__ = ", ".join(author_dict.keys())
__author_email__ = ", ".join(s for _, s in author_dict.items())
