# streamlitapp.py
import streamlit as st
import sqlite3
from datetime import datetime

# Initialize database
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            last_login DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None

# Database helper functions
def get_user(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    conn.close()
    return user

def create_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute(
            'INSERT INTO users (username, password) VALUES (?, ?)',
            (username, password)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def update_last_login(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute(
        'UPDATE users SET last_login = ? WHERE username = ?',
        (datetime.now(), username)
    )
    conn.commit()
    conn.close()

# Initialize the database
init_db()

# Login function
def login():
    st.title("Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            user = get_user(username)
            if user and password == user[2]:  # Compare plain text
                st.session_state.authenticated = True
                st.session_state.username = username
                update_last_login(username)
                st.success("Logged in successfully!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")

# Registration function
def register():
    st.title("Register")
    
    with st.form("register_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit_button = st.form_submit_button("Register")
        
        if submit_button:
            if password != confirm_password:
                st.error("Passwords don't match")
            else:
                if create_user(username, password):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Username already exists")

# Dashboard function
def dashboard():
    st.title(f"Welcome, {st.session_state.username}!")
    st.write("You're now logged in.")
    
    user = get_user(st.session_state.username)
    st.write(f"Account created: {user[4]}")
    st.write(f"Last login: {user[3] or 'Never'}")
    
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.experimental_rerun()

# Main app logic
def main():
    if not st.session_state.authenticated:
        tab1, tab2 = st.tabs(["Login", "Register"])
        with tab1:
            login()
        with tab2:
            register()
    else:
        dashboard()

if __name__ == "__main__":
    main()
