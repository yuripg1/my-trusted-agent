from ai.core import Ai
from chat import chat_loop
from database import close_db_connection, init_db, open_db_connection
from environment import Environment
from ui.core import Ui


def main() -> None:
    environment = Environment()
    db_connection = open_db_connection(environment.db_path)
    init_db(db_connection)
    ai = Ai(environment)
    ui = Ui(environment)
    chat_loop(environment, db_connection, ai, ui)
    close_db_connection(db_connection)


if __name__ == "__main__":
    main()
