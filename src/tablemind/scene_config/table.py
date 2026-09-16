"""Geometry constants for the TABLEMIND dinner-table workspace."""

TABLE_TOP_Z = 0.76

# MuJoCo box sizes are half-extents.
#
# Physical tabletop:
#
# width  = 1.20 m
# depth  = 1.00 m
# height = 0.12 m
#
TABLE_SIZE = (0.60, 0.50, 0.06)


# ------------------------------------------------------------------
# ACTIVE PLATES
# ------------------------------------------------------------------

# Plate radius = 0.115 m
# Plate half-height = 0.012 m
#
# The bottom of each plate therefore rests at:
#
# TABLE_TOP_Z + 0.012 - 0.012 = TABLE_TOP_Z

PLATE_POSITIONS = (
    ((-0.28, 0.00, TABLE_TOP_Z + 0.012)),
    (0.28, 0.00, TABLE_TOP_Z + 0.012),
)


# ------------------------------------------------------------------
# ACTIVE GLASSES
# ------------------------------------------------------------------

# Glass radius = 0.038 m
# Glass half-height = 0.075 m
#
# The bottom of each glass therefore rests at:
#
# TABLE_TOP_Z + 0.075 - 0.075 = TABLE_TOP_Z

GLASS_POSITIONS = (
    (-0.40, -0.14, TABLE_TOP_Z + 0.075),
    (0.40, -0.14, TABLE_TOP_Z + 0.075),
)


# ------------------------------------------------------------------
# RESERVE OBJECTS
# ------------------------------------------------------------------
#
# These objects exist in the MuJoCo model but are initially kept
# below the table workspace.
#
# They become available when a conversational modification increases
# the required quantity.
#
# plate_3 -> reserve plate
# glass_3 -> reserve glass
#
# The reserve Z values are intentionally well below the tabletop so
# the perception layer can distinguish unavailable objects from
# active table objects.

RESERVE_PLATE_POSITIONS = (
    (0.00, 0.00, 0.20),
)

RESERVE_GLASS_POSITIONS = (
    (0.00, 0.00, 0.20),
)
