import subprocess
import sys

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
    thread = threading.Thread(target=call_board)
    thread.start()
    pack_board_buttons()


def load_def_position():
    def_position = chess.Board()
    # def_position.set_board_fen('rnbqk3/pppp1pPp/8/8/8/8/PPPPPP1P/RNBQKBNR')
    # def_position.push_uci('g7g8q')
    board.load_position(def_position)


def pack_board_buttons():
    load_start_position.pack(pady=5, side="left", padx=10)
    apply_size_button.pack(side="right", padx=10)
    change_piece_size_entry.pack(side="right")
    undo_move_button.pack()


size_variable = tk.DoubleVar(value=1.15)


def apply_size():
    board.change_piece_size(float(change_piece_size_entry.get()))


def undo_move():
    board.undo_move()


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

spawn_board.pack(pady=10, padx=10)

# Definición de constantes y funciones para la lógica del juego
PAWNV = 10
KNIGHTV = 32
BISHOPV = 34
ROOKV = 50
QUEENV = 90
KINGV = 200

PAWN_PST = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    50,
    50,
    50,
    50,
    50,
    50,
    50,
    50,
    10,
    10,
    20,
    30,
    30,
    20,
    10,
    10,
    5,
    5,
    10,
    25,
    25,
    10,
    5,
    5,
    0,
    0,
    0,
    20,
    20,
    0,
    0,
    0,
    5,
    -5,
    -10,
    0,
    0,
    -10,
    -5,
    5,
    5,
    10,
    10,
    -20,
    -20,
    10,
    10,
    5,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]

KNIGHT_PST = [
    -50,
    -40,
    -30,
    -30,
    -30,
    -30,
    -40,
    -50,
    -40,
    -20,
    0,
    0,
    0,
    0,
    -20,
    -40,
    -30,
    0,
    10,
    15,
    15,
    10,
    0,
    -30,
    -30,
    5,
    15,
    20,
    20,
    15,
    5,
    -30,
    -30,
    0,
    15,
    20,
    20,
    15,
    0,
    -30,
    -30,
    5,
    10,
    15,
    15,
    10,
    5,
    -30,
    -40,
    -20,
    0,
    5,
    5,
    0,
    -20,
    -40,
    -50,
    -40,
    -30,
    -30,
    -30,
    -30,
    -40,
    -50,
]

BISHOP_PST = [
    -20,
    -10,
    -10,
    -10,
    -10,
    -10,
    -10,
    -20,
    -10,
    0,
    0,
    0,
    0,
    0,
    0,
    -10,
    -10,
    0,
    5,
    10,
    10,
    5,
    0,
    -10,
    -10,
    5,
    5,
    10,
    10,
    5,
    5,
    -10,
    -10,
    0,
    10,
    10,
    10,
    10,
    0,
    -10,
    -10,
    10,
    10,
    10,
    10,
    10,
    10,
    -10,
    -10,
    5,
    0,
    0,
    0,
    0,
    5,
    -10,
    -20,
    -10,
    -10,
    -10,
    -10,
    -10,
    -10,
    -20,
]

ROOK_PST = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    5,
    10,
    10,
    10,
    10,
    10,
    10,
    5,
    -5,
    0,
    0,
    0,
    0,
    0,
    0,
    -5,
    -5,
    0,
    0,
    0,
    0,
    0,
    0,
    -5,
    -5,
    0,
    0,
    0,
    0,
    0,
    0,
    -5,
    -5,
    0,
    0,
    0,
    0,
    0,
    0,
    -5,
    -5,
    0,
    0,
    0,
    0,
    0,
    0,
    -5,
    0,
    0,
    0,
    5,
    5,
    0,
    0,
    0,
]

QUEEN_PST = [
    -20,
    -10,
    -10,
    -5,
    -5,
    -10,
    -10,
    -20,
    -10,
    0,
    0,
    0,
    0,
    0,
    0,
    -10,
    -10,
    0,
    5,
    5,
    5,
    5,
    0,
    -10,
    -5,
    0,
    5,
    5,
    5,
    5,
    0,
    -5,
    0,
    0,
    5,
    5,
    5,
    5,
    0,
    -5,
    -10,
    5,
    5,
    5,
    5,
    5,
    0,
    -10,
    -10,
    0,
    5,
    0,
    0,
    0,
    0,
    -10,
    -20,
    -10,
    -10,
    -5,
    -5,
    -10,
    -10,
    -20,
]

KING_MG_PST = [
    -30,
    -40,
    -40,
    -50,
    -50,
    -40,
    -40,
    -30,
    -30,
    -40,
    -40,
    -50,
    -50,
    -40,
    -40,
    -30,
    -30,
    -40,
    -40,
    -50,
    -50,
    -40,
    -40,
    -30,
    -30,
    -40,
    -40,
    -50,
    -50,
    -40,
    -40,
    -30,
    -20,
    -30,
    -30,
    -40,
    -40,
    -30,
    -30,
    -20,
    -10,
    -20,
    -20,
    -20,
    -20,
    -20,
    -20,
    -10,
    20,
    20,
    0,
    0,
    0,
    0,
    20,
    20,
    20,
    30,
    10,
    0,
    0,
    10,
    30,
    20,
]

KING_EG_PST = [
    -50,
    -40,
    -30,
    -20,
    -20,
    -30,
    -40,
    -50,
    -30,
    -20,
    -10,
    0,
    0,
    -10,
    -20,
    -30,
    -30,
    -10,
    20,
    30,
    30,
    20,
    -10,
    -30,
    -30,
    -10,
    30,
    40,
    40,
    30,
    -10,
    -30,
    -30,
    -10,
    30,
    40,
    40,
    30,
    -10,
    -30,
    -30,
    -10,
    20,
    30,
    30,
    20,
    -10,
    -30,
    -30,
    -30,
    0,
    0,
    0,
    0,
    -30,
    -30,
    -50,
    -30,
    -30,
    -30,
    -30,
    -30,
    -30,
    -50,
]

IA_PLAYS_AS = chess.BLACK


def invert_pst(pst):
    matrix = np.array(pst).reshape(8, 8)
    matrix_invertida = np.flipud(matrix)
    inverted = matrix_invertida.flatten().tolist()
    return inverted


def evaluate_piece(piece, square, endgame=False, pst=True):
    if piece.color == chess.BLACK:
        if piece.piece_type == chess.PAWN:
            if pst:
                return PAWN_PST[square] + PAWNV
            return PAWNV
        elif piece.piece_type == chess.KNIGHT:
            if pst:
                return KNIGHT_PST[square] + KNIGHTV
            return KNIGHTV
        elif piece.piece_type == chess.BISHOP:
            if pst:
                return BISHOP_PST[square] + BISHOPV
            return BISHOPV
        elif piece.piece_type == chess.ROOK:
            if pst:
                return ROOK_PST[square] + ROOKV
            return ROOKV
        elif piece.piece_type == chess.QUEEN:
            if pst:
                return QUEEN_PST[square] + QUEENV
            return QUEENV
        elif piece.piece_type == chess.KING:
            if pst:
                if endgame:
                    return KING_EG_PST[square] + KINGV
                return KING_MG_PST[square] + KINGV
            return 0
    else:
        if piece.piece_type == chess.PAWN:
            if pst:
                return invert_pst(PAWN_PST)[square] + PAWNV
            return PAWNV
        elif piece.piece_type == chess.KNIGHT:
            if pst:
                return invert_pst(KNIGHT_PST)[square] + KNIGHTV
            return KNIGHTV
        elif piece.piece_type == chess.BISHOP:
            if pst:
                return invert_pst(BISHOP_PST)[square] + BISHOPV
            return BISHOPV
        elif piece.piece_type == chess.ROOK:
            if pst:
                return invert_pst(ROOK_PST)[square] + ROOKV
            return ROOKV
        elif piece.piece_type == chess.QUEEN:
            if pst:
                return invert_pst(QUEEN_PST)[square] + QUEENV
            return QUEENV
        elif piece.piece_type == chess.KING:
            if pst:
                if endgame:
                    return invert_pst(KING_EG_PST)[square] + KINGV
                return invert_pst(KING_MG_PST)[square] + KINGV
            return 0


def evaluate(board: chess.Board, log=False) -> float:
    global white_turn
    if board.is_checkmate():
        if white_turn:
            return -1000
        else:
            return 1000
    if (
        board.is_fifty_moves()
        or board.is_fivefold_repetition()
        or board.is_insufficient_material()
        or board.is_repetition()
        or board.is_stalemate()
        or board.is_seventyfive_moves()
    ):
        return 0

    total_material = evaluate_by_material(
        board, chess.WHITE, False, False
    ) + evaluate_by_material(board, chess.BLACK, False, False)
    endgame = total_material < 300
    if log:
        print(total_material)

    score = evaluate_by_material(board, chess.WHITE, endgame)
    score -= evaluate_by_material(board, chess.BLACK, endgame)
    return score


def squareset_to_quantity(squareset: chess.SquareSet) -> int:
    squareset = str(squareset)
    squareset = squareset.replace(" ", "")
    squareset = squareset.replace(".", "")
    squareset = squareset.replace("\n", "")
    squareset = len(squareset)
    return squareset


def evaluate_by_material(board: chess.Board, color: bool, endgame=False, pst=True):
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = evaluate_piece(piece, square, endgame, pst)
            if color == piece.color:
                score += value
    return score


def color_material(color: bool, board: chess.Board) -> int:
    return (
        squareset_to_quantity(board.pieces(chess.PAWN, color)) * PAWNV
        + squareset_to_quantity(board.pieces(chess.KNIGHT, color)) * KNIGHTV
        + squareset_to_quantity(board.pieces(chess.BISHOP, color)) * BISHOPV
        + squareset_to_quantity(board.pieces(chess.ROOK, color)) * ROOKV
        + squareset_to_quantity(board.pieces(chess.QUEEN, color)) * QUEENV
    )


# @cache
def minimax(
    position: chess.Board, depth: int, alpha: float, beta: float, white_turn: bool
):
    if depth == 0 or position.is_game_over():
        return evaluate(position), None

    best_move = None

    if white_turn:
        best = -float("inf")
        for move in position.legal_moves:
            position.push(move)  # Make the move
            val, _ = minimax(
                position, depth - 1, alpha, beta, not white_turn
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
                position, depth - 1, alpha, beta, not white_turn
            )  # Evaluate move
            position.pop()  # Undo the move

            if val < best:
                best = val
                best_move = move

            beta = min(beta, best)
            if beta <= alpha:
                break  # Alpha cutoff, prune the remaining moves

    return best, best_move


class Board:

    def __init__(self):
        global screen, selected_square, clock, WIDTH, HEIGHT, FPS, first, square_dim, current_board, piece_size, highlight, white_turn, ai_highlight, depth

        # Inicialización
        pygame.init()
        pygame.font.init()
        tick = 0

        # Ancho y alto
        WIDTH, HEIGHT = 725, 725

        square_dim = WIDTH / 8
        piece_size = square_dim * 1.15

        # Fps
        FPS = 60

        empty_board = chess.Board.empty()
        current_board = empty_board

        # Ventana de Pygame
        screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.SRCALPHA)
        highlight = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        holding_piece = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.display.set_caption(
            os.path.basename(os.path.dirname(os.path.abspath(__file__)))
        )

        # Reloj de Pygame
        clock = pygame.time.Clock()

        running = True

        # Funciones auxiliares para el dibujo y manejo del tablero
        def render_text(
            content: str,
            font: str,
            size: int,
            color,
            surface,
            bold=False,
            italic=False,
            antialias=False,
            center_x=0,
            center_y=0,
            x=0,
            y=0,
        ):
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
        start = None
        end = None
        white_turn = True
        start_highlight = None
        next = -1
        ai_highlight = ((-1, -1), (-1, -1))
        move_highlight = ai_highlight
        holding = None
        depth = 4

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    old = start
                    start = selected_square
                    if end == "Nan":
                        start = old
                        end = selected_square
                    else:
                        start_square = (
                            self.san_idx(selected_square)[0]
                            + self.san_idx(selected_square)[1] * 8
                        )
                        old = current_board
                        # current_board = ('\n'.join(str(current_board).strip().split('\n')[::-1]))
                        current_board = self.position_to_representation(current_board)
                        if not current_board[start_square].islower():
                            holding = str(current_board[start_square]).capitalize()
                        else:
                            holding = current_board[start_square].lower()
                        images = self.load_pieces()
                        current_board = old
                    start_highlight = self.san_idx(start)
                if event.type == pygame.MOUSEBUTTONUP:
                    holding = None
                    end = selected_square
                    if selected_square == start:
                        end = "Nan"
                    else:

                        try:
                            if IA_PLAYS_AS:
                                if not white_turn:
                                    os.system("cls")
                                    promotion = ""
                                    # print(start[1], end=", ")
                                    # print(end[1])
                                    if self.position_to_representation(current_board)[
                                        (
                                            self.san_idx(start)[0]
                                            + self.san_idx(start)[1] * 8
                                        )
                                    ].lower() == "p" and (
                                        start[1],
                                        end[1] == 7,
                                        8 or start[1],
                                        end[1] == 2,
                                        1,
                                    ):
                                        print("tried promotion")
                                        promotion = input(
                                            "You will promote a pawn, [q, r, b, n]"
                                        )
                                    current_board.push_uci(
                                        str(str(start) + str(end) + promotion)
                                    )
                                    ai_highlight = ((-1, -1), (-1, -1))
                                    print(f"AI evaluation: {evaluate(current_board)}")
                                    next = 2
                            else:
                                if white_turn:
                                    os.system("cls")
                                    promotion = ""
                                    # print(start[1], end=", ")
                                    # print(end[1])
                                    if self.position_to_representation(current_board)[
                                        (
                                            self.san_idx(start)[0]
                                            + self.san_idx(start)[1] * 8
                                        )
                                    ].lower() == "p" and (
                                        (start[1] == 7 and end[1] == 8)
                                        or (start[1] == 2 and end[1] == 1)
                                    ):
                                        promotion = input(
                                            "You will promote a pawn, [q, r, b, n]"
                                        )
                                    current_board.push_uci(
                                        str(str(start) + str(end) + promotion)
                                    )
                                    ai_highlight = ((-1, -1), (-1, -1))
                                    print(f"AI evaluation: {evaluate(current_board)}")
                                    next = 2
                        except chess.IllegalMoveError:
                            print(f"Move {str(str(start) + str(end))} is illegal")
                        finally:
                            ...
                            # print(f'send move: {str(str(start) + str(end))}')

            screen.fill("purple")
            highlight.fill((0, 0, 0, 0))
            holding_piece.fill((0, 0, 0, 0))

            if next > 0:
                next -= 1
            if next == 0:
                white_turn = not white_turn
                next = -1

            top_lefts = self.get_top_lefts()

            draw_board()
            ixiy = self.highlight_mouse_square()
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
            if ai_highlight != ((-1, -1), (-1, -1)):
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
            self.draw_position(current_board)
            if holding:
                images = self.load_pieces()
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
                selected_square = self.san_idx((ix, iy))
            # render_text(str(white_turn), 'Consolas', 24, 'green', screen, True)

            ai_uci = self.ia_play()
            if ai_uci and ai_uci != -1:
                ai_uci = str(ai_uci)
                ai_1 = self.san_idx(ai_uci[:2])
                ai_2 = self.san_idx(ai_uci[-2:])
                # print(f'{ai_uci}: {ai_1}, {ai_2}')
                ai_highlight = (ai_1, ai_2)
                start_highlight = None
                ai_uci = None
                print(f"AI evaluation: {evaluate(current_board)}")
            elif ai_uci == -1:
                ai_uci = None
                current_board = chess.Board
                white_turn = True

            fps = clock.tick(FPS)
            pygame.display.flip()

        pygame.quit()

    # Definición de funciones de la clase Board
    def load_pieces(self):
        global square_dim
        global piece_size
        images = {
            "P": pygame.image.load("/res/wp.png"),
            "R": pygame.image.load("/res/wr.png"),
            "N": pygame.image.load("/res/wn.png"),
            "B": pygame.image.load("/res/wb.png"),
            "Q": pygame.image.load("/res/wq.png"),
            "K": pygame.image.load("/res/wk.png"),
            "p": pygame.image.load("/res/p.png"),
            "r": pygame.image.load("/res/r.png"),
            "n": pygame.image.load("/res/n.png"),
            "b": pygame.image.load("/res/b.png"),
            "q": pygame.image.load("/res/q.png"),
            "k": pygame.image.load("/res/k.png"),
        }
        for idx, val in images.items():
            key: pygame.surface.Surface = pygame.transform.scale(
                val, (piece_size, piece_size)
            )
            images[idx] = key
        return images

    def position_to_representation(self, pos):
        pos = str(pos)
        pos = pos.replace(" ", "")
        pos = pos.replace("\n", "")
        return pos

    def draw_position(self, position):
        global square_dim
        global screen
        global piece_size
        global top_lefts
        position = self.position_to_representation(position)
        imgs = self.load_pieces()
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

    def ia_play(self):
        global white_turn
        global depth
        if IA_PLAYS_AS:
            if white_turn:
                val = minimax(
                    current_board, depth, -float("inf"), float("inf"), white_turn
                )
                if isinstance(val, int):
                    print("Checkmate")
                    self.load_position(chess.Board())
                if not isinstance(val, int):
                    self.make_move(val[1])
                    white_turn = not white_turn
                return val[1]
        else:
            if not white_turn:
                val = minimax(
                    current_board, depth, -float("inf"), float("inf"), white_turn
                )
                if isinstance(val, int):
                    print("Checkmate")
                    self.load_position(chess.Board())
                if not isinstance(val, int):
                    self.make_move(val[1])
                    white_turn = not white_turn
                try:
                    return val[1]
                except TypeError:
                    return -1

    def load_position(self, pos):
        global current_board
        current_board = pos

    def change_piece_size(self, size):
        global piece_size
        global square_dim
        piece_size = size * square_dim

    def make_move(self, move):
        if move:
            global current_board
            current_board.push_uci(str(move))
        else:
            print("Checkmate")

    def highlight_mouse_square(self):
        global top_lefts
        global square_dim
        global screen
        global highlight
        self.get_top_lefts()
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

    def get_top_lefts(self):
        global top_lefts
        top_lefts = []
        i = -1
        for rank in range(8):
            for col in range(8):
                i += 1
                top_lefts.append(((i % 8 * square_dim), (i // 8 * square_dim)))
        return top_lefts

    def san_idx(self, input):
        letters = "abcdefgh"
        if isinstance(input, str):
            if len(input) == 2:
                ix = ord(input[0]) - ord("a")
                iy = (int(input[1]) * -1) + 8
                return ix, iy
        else:
            return str(f"{letters[input[0]]}{-1 * ((input[1] + 1) - 9)}")

    def undo_move(self):
        global current_board
        global ai_highlight
        current_board.pop()
        current_board.pop()
        ai_highlight = ((-1, -1), (-1, -1))

    def set_depth(self, val):
        global depth
        depth = val


# Iniciar la aplicación
window.mainloop()
