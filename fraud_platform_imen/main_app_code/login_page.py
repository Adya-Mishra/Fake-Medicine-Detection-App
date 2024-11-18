# login_page.py

import codecs
import streamlit as st
import pymysql  
import bcrypt
import base64
from streamlit_extras.switch_page_button import switch_page

# MySQL Connection using PyMySQL
db = pymysql.connect(
    host="localhost",
    user="root",
    password="Abc@1234567890",
    database="fake_pharma_users"
)

# ----------------------------------------- FUNCTIONS -----------------------------------------

def check_password(provided_password, stored_password_hash):
    """Verify password using bcrypt."""
    return bcrypt.checkpw(provided_password, stored_password_hash)

def to_bytes(s):
    """Convert string to bytes, handling different input types."""
    if isinstance(s, bytes):
        return s
    elif isinstance(s, str):
        return codecs.encode(s, 'utf-8')
    else:
        raise TypeError("Expected bytes or string, but got %s." % type(s))

def validate_login(email, password):
    """Validate the user's login credentials."""
    cursor = db.cursor()
    query = "SELECT password FROM users WHERE email = %s"
    cursor.execute(query, (email,))
    stored_password_hash = cursor.fetchone()

    if stored_password_hash:
        # Convert stored password hash to bytes if necessary
        stored_password_hash = stored_password_hash[0].encode('utf-8')
        return check_password(to_bytes(password), stored_password_hash)
    return False

def create_account(email, password):
    """Create a new user account with a hashed password."""
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    cursor = db.cursor()
    query = "INSERT INTO users (email, password) VALUES (%s, %s)"
    cursor.execute(query, (email, hashed_password))
    db.commit()

def get_base64(bin_file):
    """Encode file in base64 format."""
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    """Set background image for the Streamlit app."""
    bin_str = get_base64(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
    background-image: url("data:image/png;base64,{bin_str}");
    background-size: cover;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# ----------------------------------------- SESSION STATE -----------------------------------------

# Initialize session state for login
if "user_logged_in" not in st.session_state:
    st.session_state.user_logged_in = False

# ----------------------------------------- LOGIN PAGE -----------------------------------------

st.title("Fake Medicine Detection App")
set_background(r'fraud_platform_imen\widgets\medicine-capsules.png')
st.write("Welcome! Please log in or sign up to continue:")

choice = st.radio("Select choice", ["Login", "Sign Up"])

# Sign-Up Option
if choice == "Sign Up":
    new_email = st.text_input("New Email")
    new_password = st.text_input("New Password", type="password")
    if st.button("Sign Up"):
        try:
            create_account(new_email, new_password)
            st.success("Account created successfully! Now you can log in.")
        except pymysql.MySQLError as err:  # Update to handle PyMySQL errors
            st.error(f"Error: {err}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# Login Option
elif choice == "Login":
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Log In"):
        if validate_login(email, password):
            st.session_state.user_logged_in = True
            st.success("Login Successful!")
            switch_page("main")  # Redirect to main page after successful login
        else:
            st.warning("Incorrect email or password. Please try again.")

# Optional: Log out button if the user is already logged in
if st.session_state.user_logged_in:
    if st.button("Logout"):
        st.session_state.user_logged_in = False
        st.success("Logged out successfully.")
