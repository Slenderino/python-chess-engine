import subprocess
import sys
import random
import time
import math

try:
    import tkinter as tk
    import threading
    import chess
    import numpy as np
    import pygame
    import os
    from chess.engine import EngineError
    import ttkbootstrap as ttk
    from ttkbootstrap import Style
    from functools import cache
    import queue
except ModuleNotFoundError as e:  # A module was not found
    print(f"{e.name} not found. Installing module...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
    )
    # Try imports again after installation
    try:
        import tkinter as tk
        import threading
        import chess
        import numpy as np
        import pygame
        import os
        from chess.engine import EngineError
        import ttkbootstrap as ttk
        from ttkbootstrap import Style
        from functools import cache
        import queue
    except ModuleNotFoundError:
        print("Error at importing necessary modules.")
        sys.exit(1)

# Main window configuration
window = ttk.Window()
window.title("Manager")
window.geometry("700x95+482+5")
window.configure(bg="#2F433D")

# Style for buttons
style = Style()
style.configure("TButton", background="#2F433D", foreground="white")

# Declares if is the first time opening the board
first = True

loaded = False

def new_window(message, result_queue):
    """
    Creates a new window.

    :param message: The message the window displays

    :param result_queue: Variable to put the result in

    :return: No return since the result is already stored
    """

    # Configuring window
    new_win = ttk.Window()
    new_win.title("Manager")
    new_win.geometry("700x95+482+5")
    new_win.configure(bg="#2F433D")

    # Entry variable
    entry_var = tk.StringVar()

    def close():
        """
        Simple method to close the window
        :return:
        No return since it destroys the window
        """
        result_queue.put(entry_var.get())  # Put the value in the queue
        new_win.destroy()  # Destroy the window

    # Displays the message
    label = ttk.Label(master=new_win, text=message, font="Consolas 15")
    label.pack()

    # Entry for user input
    entry = ttk.Entry(master=new_win, textvariable=entry_var)
    entry.pack()

    # Close button
    okay = ttk.Button(master=new_win, text="Ok", command=close)
    okay.pack()

    new_win.mainloop()  # Initiates the window


# Board-related functions
def call_board():
    """
    Sets the board variable to Board()
    :return: Returns the board object
    """
    global board
    board = Board()
    return Board


def new_board_thread():
    """
    Creates a new Board() thread
    :return: None
    """
    global thread
    global first
    if first:
        call_board()
        first = False
        new_board_thread()
        return
    try:
        thread = threading.Thread(target=call_board)
    except Exception as err:
        print(err)
    thread.start()
    pack_board_buttons()


def load_def_position():
    global white_turn
    def_position = chess.Board()
    # def_position.set_board_fen('rnbqk3/ppppppPp/8/8/8/8/PPPPPP1P/RNBQKBNR')
    # def_position.push_uci('g7g8q')
    load_position(def_position)
    loaded = True
    white_turn = True


def pack_board_buttons():
    load_start_position.pack(pady=5, side="left", padx=10)
    apply_size_button.pack(side="right", padx=10)
    change_piece_size_entry.pack(side="right")
    undo_move_button.pack()
    ai_button.pack()


size_variable = tk.DoubleVar(value=1.15)


def apply_size():
    change_piece_size(float(change_piece_size_entry.get()))


def undo_move():
    undo_move()

def ai_button():
    ai_play_move()
    print('ai')


# Botones de la interfaz
spawn_board = ttk.Button(
    master=window,
    command=new_board_thread,
    text="Spawn board",
    width=150,
    style="TButton",
)
load_start_position = ttk.Button(
    master=window, command=load_def_position, text="Load Default Position", width=85
)
change_piece_size_entry = ttk.Entry(master=window, width=5, textvariable=size_variable)
apply_size_button = ttk.Button(
    master=window, width=10, text="Apply", command=apply_size, style="TButton"
)
undo_move_button = ttk.Button(
    master=window, width=15, text="Undo Move", command=undo_move, style="TButton"
)

ai_button = ttk.Button(master=window, command=ai_button, text='AI', width=15, style="TButton")

spawn_board.pack(pady=10, padx=10)

current_time = time.strftime("%d-%m-%y_%H-%M")

# Definición de constantes y funciones para la lógica del juego
PAWNV = 10
KNIGHTV = 32
BISHOPV = 34
ROOKV = 50
QUEENV = 90
KINGV = 200

PAWN_PST = [
     0,  0,   0,   0,   0,   0,  0,  0,
    50, 50,  50,  50,  50,  50, 50, 50,
    10, 10,  20,  30,  30,  20, 10, 10,
     5,  5,  10,  25,  25,  10,  5,  5,
     0,  0,   0,  20,  20,   0,  0,  0,
     5, -5, -10,   0,   0, -10, -5,  5,
     5, 10,  10, -30, -30,  10, 10,  5,
     0,  0,   0,   0,   0,   0,  0,  0,
]

KNIGHT_PST = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

BISHOP_PST = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

ROOK_PST = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]

QUEEN_PST = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10,   0,   0,  0,  0,   0,   0, -10,
    -10,   0,   5,  5,  5,   5,   0, -10,
    -5,    0,   5,  5,  5,   5,   0, -5,
     0,    0,   5,  5,  5,   5,   0, -5,
    -10,   5,   5,  5,  5,   5,   0, -10,
    -10,   0,   5,  0,  0,   0,   0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
]

KING_MG_PST = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20,
]

KING_EG_PST = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
]

IA_PLAYS_AS = chess.WHITE


def invert_pst(pst):
    matrix = np.array(pst).reshape(8, 8)
    matrix_invertida = np.flipud(matrix)
    inverted = matrix_invertida.flatten().tolist()
    return inverted

POSITIONAL_INFLUENCE: float = 0.2

def evaluate_piece(piece, square, endgame=False, pst=True):
    if piece.color == chess.BLACK:
        if piece.piece_type == chess.PAWN:
            if pst:
                return PAWN_PST[square] * POSITIONAL_INFLUENCE + PAWNV
            return PAWNV
        elif piece.piece_type == chess.KNIGHT:
            if pst:
                return KNIGHT_PST[square] * POSITIONAL_INFLUENCE + KNIGHTV
            return KNIGHTV
        elif piece.piece_type == chess.BISHOP:
            if pst:
                return BISHOP_PST[square] * POSITIONAL_INFLUENCE + BISHOPV
            return BISHOPV
        elif piece.piece_type == chess.ROOK:
            if pst:
                return ROOK_PST[square] * POSITIONAL_INFLUENCE + ROOKV
            return ROOKV
        elif piece.piece_type == chess.QUEEN:
            if pst:
                return QUEEN_PST[square] * POSITIONAL_INFLUENCE + QUEENV
            return QUEENV
        elif piece.piece_type == chess.KING:
            if pst:
                if endgame:
                    return KING_EG_PST[square] * POSITIONAL_INFLUENCE + KINGV
                return KING_MG_PST[square] * POSITIONAL_INFLUENCE + KINGV
            return 0
    else:
        if piece.piece_type == chess.PAWN:
            if pst:
                return invert_pst(PAWN_PST)[square] * POSITIONAL_INFLUENCE + PAWNV
            return PAWNV
        elif piece.piece_type == chess.KNIGHT:
            if pst:
                return invert_pst(KNIGHT_PST)[square] * POSITIONAL_INFLUENCE + KNIGHTV
            return KNIGHTV
        elif piece.piece_type == chess.BISHOP:
            if pst:
                return invert_pst(BISHOP_PST)[square] * POSITIONAL_INFLUENCE + BISHOPV
            return BISHOPV
        elif piece.piece_type == chess.ROOK:
            if pst:
                return invert_pst(ROOK_PST)[square] * POSITIONAL_INFLUENCE + ROOKV
            return ROOKV
        elif piece.piece_type == chess.QUEEN:
            if pst:
                return invert_pst(QUEEN_PST)[square] * POSITIONAL_INFLUENCE + QUEENV
            return QUEENV
        elif piece.piece_type == chess.KING:
            if pst:
                if endgame:
                    return invert_pst(KING_EG_PST)[square] * POSITIONAL_INFLUENCE + KINGV
                return invert_pst(KING_MG_PST)[square] * POSITIONAL_INFLUENCE + KINGV
            return 0


def evaluate(position: chess.Board, log=False) -> float:
    global white_turn
    if position.is_checkmate():
        if position.is_attacked_by(chess.BLACK, position.king(chess.WHITE)): print('AI has found checkmate for black'); return -1000
        print('AI has found checkmate for white')
        return 1000
    if (
        position.is_fifty_moves()
        or position.is_fivefold_repetition()
        or position.is_insufficient_material()
        or position.is_repetition()
        or position.is_stalemate()
        or position.is_seventyfive_moves()
    ):
        return 0

    total_material = evaluate_by_material(
        position, chess.WHITE, False, False
    ) + evaluate_by_material(position, chess.BLACK, False, False)
    endgame = total_material < 300
    if log:
        print(total_material)

    score = evaluate_by_material(position, chess.WHITE, endgame)
    score -= evaluate_by_material(position, chess.BLACK, endgame)
    return score


def squareset_to_quantity(squareset: chess.SquareSet) -> int:
    squareset = str(squareset)
    squareset = squareset.replace(" ", "")
    squareset = squareset.replace(".", "")
    squareset = squareset.replace("\n", "")
    squareset = len(squareset)
    return squareset


def evaluate_by_material(position: chess.Board, color: bool, endgame=False, pst=True):
    score = 0
    for square in chess.SQUARES:
        piece = position.piece_at(square)
        if piece:
            value = evaluate_piece(piece, square, endgame, pst)
            if color == piece.color:
                score += value
    return score


def color_material(color: bool, position: chess.Board) -> int:
    return (
            squareset_to_quantity(position.pieces(chess.PAWN, color)) * PAWNV
            + squareset_to_quantity(position.pieces(chess.KNIGHT, color)) * KNIGHTV
            + squareset_to_quantity(position.pieces(chess.BISHOP, color)) * BISHOPV
            + squareset_to_quantity(position.pieces(chess.ROOK, color)) * ROOKV
            + squareset_to_quantity(position.pieces(chess.QUEEN, color)) * QUEENV
    )


# @cache
def minimax(
    position: chess.Board, profundity: int, alpha: float, beta: float, white_turn: bool
):
    temp = chess.Board()
    temp.push_san('e4')
    temp.push_san('e5')
    temp.push_san('Nf3')
    temp.push_san('Nc6')
    if position == temp and random.randint(0, 1) == 0: return 0, 'b8c6'
    if profundity == 0 or position.is_game_over():
        return evaluate(position), None

    best_move = None

    if white_turn:
        best = -float("inf")
        for move in position.legal_moves:
            position.push(move)  # Make the move
            val, _ = minimax(
                position, profundity - 1, alpha, beta, not white_turn
            )  # Evaluate move
            position.pop()  # Undo the move

            if val > best:
                best = val
                best_move = move

            alpha = max(alpha, best)
            if beta <= alpha:
                break  # Beta cutoff, prune the remaining moves

    else:
        best = float("inf")
        for move in position.legal_moves:
            position.push(move)  # Make the move
            val, _ = minimax(
                position, profundity - 1, alpha, beta, not white_turn
            )  # Evaluate move
            position.pop()  # Undo the move

            if val < best:
                best = val
                best_move = move

            beta = min(beta, best)
            if beta <= alpha:
                break  # Alpha cutoff, prune the remaining moves

    return best, best_move


def get_top_lefts():
    global top_lefts
    top_lefts = []
    i = -1
    for rank in range(8):
        for col in range(8):
            i += 1
            top_lefts.append(((i % 8 * square_dim), (i // 8 * square_dim)))
    return top_lefts


def set_depth(val):
    global depth
    depth = val


def undo_move():
    global current_board
    global ai_highlight
    current_board.pop()
    current_board.pop()
    ai_highlight = ((-1, -1), (-1, -1))


def highlight_mouse_square():
    global top_lefts
    global square_dim
    global screen
    global highlight
    get_top_lefts()
    pos = pygame.mouse.get_pos()
    if not (pos[0] > 0 and pos[1] > 0):
        return
    ix = int(pos[0] // square_dim)
    iy = int(pos[1] // square_dim * 8)
    x = top_lefts[ix][0]
    y = top_lefts[iy][1]
    pygame.draw.rect(
        highlight,
        (255, 0, 0, 100),
        pygame.Rect((x, y), (square_dim + 1, square_dim + 1)),
    )
    iy = iy // 8
    return ix, iy


def change_piece_size(size):
    global piece_size
    global square_dim
    piece_size = size * square_dim


def load_position(pos):
    global current_board
    current_board = pos


def position_to_representation(pos):
    pos = str(pos)
    pos = pos.replace(" ", "")
    pos = pos.replace("\n", "")
    return pos


def load_pieces():
    global square_dim
    global piece_size
    os.chdir(os.path.dirname(__file__))
    images = {
        "P": pygame.image.load("res/wp.png"),
        "R": pygame.image.load("res/wr.png"),
        "N": pygame.image.load("res/wn.png"),
        "B": pygame.image.load("res/wb.png"),
        "Q": pygame.image.load("res/wq.png"),
        "K": pygame.image.load("res/wk.png"),
        "p": pygame.image.load("res/p.png"),
        "r": pygame.image.load("res/r.png"),
        "n": pygame.image.load("res/n.png"),
        "b": pygame.image.load("res/b.png"),
        "q": pygame.image.load("res/q.png"),
        "k": pygame.image.load("res/k.png"),
    }
    for idx, val in images.items():
        key: pygame.surface.Surface = pygame.transform.scale(
            val, (piece_size, piece_size)
        )
        images[idx] = key
    return images


def draw_position(position):
    global square_dim
    global screen
    global piece_size
    global top_lefts
    position = position_to_representation(position)
    imgs = load_pieces()
    centers = []
    top_lefts = []
    i = -1
    for rank in range(8):
        for col in range(8):
            i += 1
            centers.append(
                (
                    (i % 8 * square_dim) + square_dim / 2,
                    (i // 8 * square_dim) + square_dim / 2,
                )
            )
            top_lefts.append(((i % 8 * square_dim), (i // 8 * square_dim)))
    for ind, p in enumerate(position):
        if p != ".":
            img: pygame.surface.Surface = imgs[p]
            screen.blit(
                img,
                (
                    centers[ind][0] - piece_size / 2,
                    centers[ind][1] - piece_size / 2,
                ),
            )


def ai_play_move():
    global ai_highlight
    global current_board
    global start_highlight
    ai_uci = ia_play()
    if ai_uci and ai_uci != -1:
        # ai_uci = str(ai_uci)
        ai_1 = san_idx(str(ai_uci)[:2])
        ai_2 = san_idx(str(ai_uci)[-2:])
        # print(f'{ai_uci}: {ai_1}, {ai_2}')
        ai_highlight = (ai_1, ai_2)
        start_highlight = None
        # print(ai_uci, type(ai_uci), 'ai_play_move()')
        write_to_log(ai_uci, current_board)
        ai_uci = None
        # print(f"AI evaluation: {evaluate(current_board)}")
        if abs(evaluate(current_board)) == 1000:
            trigger_checkmate(evaluate(current_board) > 0)  # If positive pass True(chess.WHITE)
        return ai_uci
    elif ai_uci == -1:
        ai_uci = None
        # current_board = chess.Board
        # white_turn = True


class Board:
    def __init__(self):
        global screen, selected_square, clock, WIDTH, HEIGHT, FPS, first, square_dim, current_board, piece_size, highlight, white_turn, ai_highlight, depth, tick, holding_piece, display_winning_sign, timer

        # Initialize pygame
        self.pygame_init()

        # Ancho y alto
        WIDTH, HEIGHT = 725, 725

        square_dim, piece_size = self.get_square_dim_and_piece_size()
        timer = True
        display_winning_sign = None

        # Fps
        FPS = 60

        # Create New Board
        empty_board = chess.Board.empty()
        current_board = empty_board

        # Ventana de Pygame
        self.get_displays()

        # Reloj de Pygame
        clock = pygame.time.Clock()

        self.running = True

        # Funciones auxiliares para el dibujo y manejo del tablero
        def render_text(content: str, font: str, size: int, color, surface, bold=False, italic=False, antialias=False, center_x=0, center_y=0, x=0, y=0):
            font = pygame.font.SysFont(font, size, bold)
            text = font.render(content, True, color)
            if center_x:
                textpos = text.get_rect(centerx=surface.get_width() / center_x + 1, y=y)
            elif center_y:
                textpos = text.get_rect(
                    x=x, centery=surface.get_height() / center_y + 1
                )
            elif center_x and center_y:
                textpos = text.get_rect(
                    centerx=surface.get_width() / (center_x + 1),
                    centery=surface.get_height() / (center_y + 1),
                )
            else:
                textpos = text.get_rect(x=x, y=y)
            surface.blit(text, textpos)

        def draw_board():
            global square_dim
            global piece_size

            if WIDTH == HEIGHT:
                square_dim = WIDTH / 8
                i = -1
                for rank in range(8):
                    for col in range(8):
                        i += 1
                        colors = [(255, 255, 255), (47, 67, 61)]
                        pygame.draw.rect(
                            screen,
                            colors[(i % 8 + i // 8) % 2],
                            pygame.Rect(
                                (i % 8 * square_dim, i // 8 * square_dim),
                                ((i % 8 + 1) * square_dim, (i // 8 + 1) * square_dim),
                            ),
                        )
            else:
                raise EngineError("Width does not equal Height")

        # Lógica principal del juego
        end, holding, next, now, start, start_highlight = self.reset_variables()
        depth = 4

        while self.running:
            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    self.running = False

                if event.type == pygame.MOUSEBUTTONDOWN:

                    old = start  # Save start
                    start = selected_square  # Update start

                    """
                    if end == "Nan":  # If there's no end this must be the end (two click move)
                        start = old; end = selected_square  # Backup start and assign end
                    else:  # There's an end and therefore this is a new click
                    """

                    start_square = (san_idx(selected_square)[0] + san_idx(selected_square)[1] * 8)  # Assign start square as current square

                    old = current_board  # Backup current_board
                    current_board = position_to_representation(current_board)  # Get representation of current board

                    holding = current_board[start_square]  # Set holding piece from the current_board's start_square
                    images = load_pieces()  # Load piece images
                    current_board = old  # Return current_board from representation

                    start_highlight = san_idx(start)  # Set highlight square

                if event.type == pygame.MOUSEBUTTONUP:
                    holding = None  # Not holding any piece anymore
                    end = selected_square  # Set currrent square as end
                    if selected_square == start: end = "Nan"; start = None  # Stopped holding in same place as start, this means is now a two click move, could alternatively add start = None to impede two click moves, which is more intuitive to the user
                    else:

                        try:
                            if IA_PLAYS_AS == chess.WHITE:
                                if not white_turn:
                                    next, now, move = self.push_user_move(current_board, end, next, now, start)
                                    move = chess.Move.from_uci(move)
                                    # print(move, type(move), 'mousebuttonup')
                                    write_to_log(move, current_board)
                                    # print(f"AI evaluation: {evaluate(current_board)}")
                            elif IA_PLAYS_AS == chess.BLACK:
                                if white_turn:
                                    next, now, move = self.push_user_move(current_board, end, next, now, start)
                                    write_to_log(move, current_board)
                                    # print(f"AI evaluation: {evaluate(current_board)}")
                            if abs(evaluate(current_board)) == 1000:
                                trigger_checkmate(evaluate(current_board) > 0)  # If positive pass True(chess.WHITE)
                        except chess.IllegalMoveError:
                            print(f"Move {str(str(start) + str(end))} is illegal")
                        finally:
                            ...
                            # print(f'send move: {str(str(start) + str(end))}')

            pygame.display.set_caption(current_board.fen())  # Set window title as current FEN

            # Screens reset
            screen.fill("purple")
            highlight.fill((0, 0, 0, 0))
            holding_piece.fill((0, 0, 0, 0))

            # Logic for toggling white_turn after amount of frames
            if next > 0: next -= 1
            if next == 0: white_turn = not white_turn; next = -1

            top_lefts = get_top_lefts()

            draw_board()
            ixiy = highlight_mouse_square()
            if start_highlight:
                pygame.draw.rect(
                    highlight,
                    (255, 0, 0, 100),
                    pygame.Rect(
                        (
                            top_lefts[start_highlight[0]][0],
                            top_lefts[start_highlight[1] * 8][1],
                        ),
                        (square_dim + 1, square_dim + 1),
                    ),
                )
            if ai_highlight != ((-1, -1), (-1, -1)):  # There is a highlight by ai
                # start_highlight = None
                pygame.draw.rect(
                    highlight,
                    (255, 113, 54, 100),
                    pygame.Rect(
                        (
                            top_lefts[ai_highlight[0][0]][0],
                            top_lefts[ai_highlight[0][1] * 8][1],
                        ),
                        (square_dim + 1, square_dim + 1),
                    ),
                )
                pygame.draw.rect(
                    highlight,
                    (255, 113, 54, 100),
                    pygame.Rect(
                        (
                            top_lefts[ai_highlight[1][0]][0],
                            top_lefts[ai_highlight[1][1] * 8][1],
                        ),
                        (square_dim + 1, square_dim + 1),
                    ),
                )
            screen.blit(highlight, (0, 0))
            draw_position(current_board)  # Draw pieces
            if holding == '.':
                holding = None
            if holding:
                images = load_pieces()
                images[holding].set_alpha(128)
                holding_piece.blit(
                    images[holding],
                    (
                        pygame.mouse.get_pos()[0] - (square_dim) / 2,
                        pygame.mouse.get_pos()[1] - (square_dim) / 2,
                    ),
                )
                images[holding].set_alpha(255)
            screen.blit(holding_piece, (0, 0))
            if ixiy:
                ix, iy = ixiy
                selected_square = san_idx((ix, iy))
            # render_text(str(white_turn), 'Consolas', 24, 'green', screen, True)

            self.ai_play_move()


            if timer: render_text(str(time.time() - now), 'Consolas', 34, 'green', screen)
            if display_winning_sign: render_text(display_winning_sign, 'Cascadia Code', 85, 'gray', screen, x=10, y=HEIGHT/2-40)

            fps = clock.tick(FPS)
            pygame.display.flip()

        pygame.quit()

    def ai_play_move(self):
        return ai_play_move()

    def push_user_move(self, current_board, end, next, now, start):
        os.system("cls")  # Clear command prompt
        promotion = ""  # Clear promotion
        move = self.push_move_with_promotion(current_board, end, promotion, start)
        next, now = self.reset_after_move()
        return next, now, move

    def push_move_with_promotion(self, current_board, end, promotion, start):
        if (position_to_representation(current_board)[
            (san_idx(start)[0] + san_idx(start)[1] * 8)].lower() == "p"  # Piece to move is a pawn
                and ((start[1] == '7' and end[1] == '8') or (start[1] == 2 and end[1] == 1)
                # Moves from rank 7 to 8 or 2 to 1 (promotion)
                )):
            promotion = self.ask_promotion(promotion)  # Ask user promotion
        current_board.push_uci(
            str(str(start) + str(end) + promotion)  # Push uci with promotion
        )
        return str(str(start) + str(end) + promotion)

    def reset_after_move(self):
        global ai_highlight
        ai_highlight = ((-1, -1), (-1, -1))  # Reset highlight
        next = 2  # Invert white_turn in 2 frames
        now = time.time()  # Reset time
        return next, now

    def ask_promotion(self, promotion):
        promotion = input(
            "You will promote a pawn, [q, r, b, n]"
        )
        return promotion

    def reset_variables(self):
        global white_turn, ai_highlight
        now = time.time()
        start = None
        end = None
        white_turn = True
        start_highlight = None
        next = -1
        ai_highlight = ((-1, -1), (-1, -1))
        move_highlight = ai_highlight
        holding = None
        return end, holding, next, now, start, start_highlight

    def get_displays(self):
        global screen, highlight, holding_piece
        screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.SRCALPHA)
        highlight = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        holding_piece = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.display.set_caption("Board")

    def get_square_dim_and_piece_size(self):
        global square_dim, piece_size, WIDTH
        square_dim = WIDTH / 8
        piece_size = square_dim * 1.15
        return square_dim, piece_size

    def pygame_init(self):
        global tick
        # Inicialización
        pygame.init()
        pygame.font.init()
        tick = 0

    # Definición de funciones de la clase Board

def ia_play():
    global white_turn
    global depth
    global current_board
    if IA_PLAYS_AS == chess.WHITE:
        if white_turn:
            val = do_minimax(current_board)
            return val[1]
    else:
        if not white_turn:
            val = do_minimax(current_board)
            try:
                return val[1]
            except TypeError:
                return -1

def make_move(move):
    if move:
        global current_board
        current_board.push_uci(str(move))
    # else:
        # print("Checkmate from makemove")
        # reset_by_checkmate()
        # load_position(chess.Board())

def san_idx(entry):
    global running
    letters = "abcdefgh"
    if isinstance(entry, str):
        if len(entry) == 2:
            ix = ord(entry[0]) - ord("a")
            iy = (int(entry[1]) * -1) + 8
            return ix, iy
    else:
        try:
            return str(f"{letters[entry[0]]}{-1 * ((entry[1] + 1) - 9)}")
        except IndexError:
            running = False

def reset_by_checkmate():
    global current_board
    ...

def do_minimax(current_position):
    global white_turn
    # print(f'val = minimax(position=\n{current_board}\n, profundity={depth}, alpha={-float("inf")}, beta={float("inf")}, white_turn={white_turn})')
    val = minimax(
        current_position, depth, -float("inf"), float("inf"), white_turn
    )
    # print(val)
    if isinstance(val, int):
        print("Checkmate from do_minimax")
        reset_by_checkmate()
        # load_position(chess.Board())
    if not isinstance(val, int):
        make_move(val[1])
        white_turn = not white_turn
    return val

def trigger_checkmate(color):
    global display_winning_sign
    global timer
    if color == chess.WHITE: winner = 'white'
    else: winner = 'black'
    timer == False
    display_winning_sign = f'{winner.capitalize()} has won the game!'

def write_to_log(movement: chess.Move, current_board: chess.Board):
    with open(f'games/{current_time}.txt', 'a') as f:
        if len(current_board.move_stack) % 2 == 1:
            f.write(f'{math.ceil((len(current_board.move_stack)/2))}. {uci_to_san(movement, current_board, movement)} | ')
        else:
            f.write(uci_to_san(movement, current_board, movement) + '\n')


def uci_to_san(uci: chess.Move, current: chess.Board, move):
    current_board = current.copy()
    # uci = chess.Move(uci[:2], uci[2:4], uci[4:])

    # Derechos de enroque
    can_castle_kingside_white = current_board.castling_rights & chess.BB_G1
    can_castle_queenside_white = current_board.castling_rights & chess.BB_C1
    can_castle_kingside_black = current_board.castling_rights & chess.BB_G8
    can_castle_queenside_black = current_board.castling_rights & chess.BB_C8

    # Enroques
    if str(uci) == 'e1g1' and can_castle_kingside_white: return 'O-O'
    if str(uci) == 'e1c1' and can_castle_queenside_white: return 'O-O-O'
    if str(uci) == 'e8g8' and can_castle_kingside_black: return 'O-O'
    if str(uci) == 'e8c8' and can_castle_queenside_black: return 'O-O-O'

    current_board.pop()

    desambiguation = 0
    start_square = uci.from_square
    end_square = uci.to_square
    promotion = uci.promotion

    # Obtiene la pieza y la pieza capturada
    piece = current_board.piece_at(start_square)
    captured_piece = current_board.piece_at(end_square)

    # Notación del movimiento
    end_square_name = chess.square_name(end_square)
    promotion_suffix = f'={chess.piece_symbol(promotion).capitalize()}' if promotion else ''

    # Manejo de la pieza
    piece_symbol = piece.symbol().capitalize() if piece.piece_type != chess.PAWN else ''
    capture_indicator = 'x' if captured_piece else ''

    # Desambiguación de la notación
    if piece_symbol == '' and capture_indicator == 'x':
        desambiguation = 1
    start_square_name = chess.square_name(start_square) if desambiguation == 2 else (
        chess.square_name(start_square)[0] if desambiguation == 1 else '')

    current_board.push(move)
    return f"{piece_symbol}{start_square_name}{capture_indicator}{end_square_name}{promotion_suffix}"



# Iniciar la aplicación
if __name__ == '__main__':
    window.mainloop()
