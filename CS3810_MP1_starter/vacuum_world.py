"""
CS3810 Mini-Project 1 - Part 1: The Environment (30 points)
============================================================

Implement the VacuumWorld class below. Do not change any method name or
signature: run_tests.py and the grading harness call them exactly as written.

Reminders from the handout (Section 1.2):

  * A state is ((row, col), frozenset_of_dirty_cells). It must be HASHABLE,
    because your searches will put states into sets and dictionaries. Use a
    tuple for the position and a frozenset for the dirt - a plain set will
    raise TypeError the first time you add it to your explored set.

  * result() must NOT mutate the state it is given. Build and return a new
    one. This is the single most common bug in this project: if you mutate,
    every state sitting in your frontier silently becomes the same state.

  * get_actions() must return actions in the canonical order below, filtered
    down to the legal ones. Illegal actions are never generated: no CLEAN in
    a clean cell, no move into a wall or an obstacle. Determinism here is
    what makes your node counts comparable with everyone else's.
"""

# The canonical action order. Filter this list; do not reorder it.
ACTION_ORDER = ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "CLEAN"]

# Row/column offsets for the four movement actions.
DELTAS = {
    "MOVE_UP": (-1, 0),
    "MOVE_DOWN": (1, 0),
    "MOVE_LEFT": (0, -1),
    "MOVE_RIGHT": (0, 1),
}


class VacuumWorld:
    """A vacuum-cleaning robot problem on a rectangular grid."""

    def __init__(self, grid, start, dirty):
        """
        Args:
            grid:  List of strings or list of lists; '#' marks an obstacle.
            start: Tuple (row, col) - the robot's starting position.
            dirty: Iterable of (row, col) tuples that begin dirty.

        Suggested attributes to set here: self.grid, self.rows, self.cols,
        self.start, self.dirty.
        """
        # raise NotImplementedError("Part 1: implement VacuumWorld.__init__")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.start = tuple(start)
        self.dirty = frozenset(dirty)

    # -- helpers ---------------------------------------------------------
    # These two are not graded directly and not called by the test harness,
    # but get_actions() and result() are much shorter if you write them.

    def in_bounds(self, pos):
        """Return True if pos is inside the grid."""
        # raise NotImplementedError("Part 1: implement in_bounds (optional)")
        row = pos[0]
        col = pos[1]
        return (0 <= row < self.rows) and (0 <= col < self.cols)

    def is_passable(self, pos):
        """Return True if pos is in bounds and is not an obstacle."""
        # raise NotImplementedError("Part 1: implement is_passable (optional)")
        row = pos[0]
        col = pos[1]

        if not self.in_bounds(pos):
            return False

        return self.grid[row][col] != "#"

    # -- problem interface -----------------------------------------------

    def initial_state(self):
        """Return the initial state as ((row, col), frozenset(dirty))."""
        # raise NotImplementedError("Part 1: implement initial_state")
        return (self.start, self.dirty)

    def is_goal(self, state):
        """Return True if the dirty set in `state` is empty."""
        # raise NotImplementedError("Part 1: implement is_goal")
        dirty = state[1]
        return len(dirty) == 0

    def get_actions(self, state):
        """Return a list of legal action names available in `state`.

        Legal means: moves stay in bounds and off obstacles, and CLEAN is
        only offered when the robot's current cell is dirty. Return the
        actions in ACTION_ORDER order.
        """
        # raise NotImplementedError("Part 1: implement get_actions")
        position = state[0]
        dirty = state[1]
        actionOrderCopy = ACTION_ORDER.copy()

        for action in DELTAS:
            delta = DELTAS[action]

            newPosition = (position[0] + delta[0], position[1] + delta[1])

            if not self.is_passable(newPosition):
                actionOrderCopy.remove(action)

        if position not in dirty:
            actionOrderCopy.remove("CLEAN")

        return actionOrderCopy

    def result(self, state, action):
        """Return the successor state produced by applying `action`.

        Must not modify `state`.
        """
        # raise NotImplementedError("Part 1: implement result")
        position = state[0]
        dirty = state[1]
        row = position[0]
        col = position[1]

        if action == "CLEAN":
            newDirty = set(dirty)
            newDirty.remove(position)
            return (position, frozenset(newDirty))
        elif action in DELTAS:
            rowChange, colChange = DELTAS[action]
            newPosition = (row + rowChange, col + colChange)
            return (newPosition, dirty)
        else:
            raise ValueError(f"Unknown action: {action}")

    def action_cost(self, state, action):
        """Return the cost of `action` in `state` (always 1 here)."""
        # raise NotImplementedError("Part 1: implement action_cost")
        return 1

    # -- debugging -------------------------------------------------------

    def render(self, state=None):
        """Return a printable picture of `state` (defaults to the initial state).

        Use the same characters as test_grids.py: '#' obstacle, 'R' robot,
        'D' dirty, '*' robot on a dirty cell, '.' clean and empty.

        This is not graded for correctness, but you will use it constantly
        while debugging. Write it first.
        """
        # raise NotImplementedError("Part 1: implement render")
        if state is None:
            state = self.initial_state()

        position = state[0]
        dirty = state[1]
        lines = []

        for row in range(self.rows):
            line = ""

            for col in range(self.cols):
                pos = (row, col)

                if self.grid[row][col] == "#":
                    line += "#"
                elif pos == position and pos in dirty:
                    line += "*"
                elif pos == position:
                    line += "R"
                elif pos in dirty:
                    line += "D"
                else:
                    line += "."

            lines.append(line)

        return "\n".join(lines)

    def __str__(self):
        return self.render()


if __name__ == "__main__":
    # Quick manual check once you have implemented the class:
    from test_grids import EXAMPLE, parse_grid

    grid, start, dirty = parse_grid(EXAMPLE)
    problem = VacuumWorld(grid, start, dirty)
    print(problem.render())
    print("initial state:", problem.initial_state())
    print("legal actions:", problem.get_actions(problem.initial_state()))
