Register & Login System 🔐

This project is a simple user authentication system that includes registration and login functionalities. Built using HTML, CSS, JavaScript, and PHP, it allows users to sign up, log in, and manage their accounts.

Features ✨

✅ User Registration – New users can create an account with their first name, last name, email, and password.

✅ User Login – Existing users can log in using their email and password.

✅ Form Validation – Ensures required fields are filled before submission.

✅ Password Encryption – Uses md5() for password hashing (⚠️ Consider upgrading to password_hash() for better security).

✅ Session Handling – Starts a session upon successful login to keep users authenticated.

✅ Error Handling – Displays messages for incorrect credentials or existing emails.

✅ Social Media Icons – Google and Facebook icons for future OAuth integration.


How It Works 🛠️

1️⃣ User Registration

1️⃣ The user fills in their First Name, Last Name, Email, and Password.

2️⃣ The form sends the data to register.php via POST.

3️⃣ The script checks if the email already exists in the database.

4️⃣ If the email is unique, the user is registered, and the password is hashed using md5().

5️⃣ The user is redirected to index.php after successful registration.


2️⃣ User Login

1️⃣ The user enters their Email & Password and submits the form.

2️⃣ The script verifies the credentials against the database.

3️⃣ If correct, a session is started, and the user is redirected to homepage.php.

4️⃣ If incorrect, an error message is displayed.



UI Design 🎨

Minimalistic & Clean Design ✅

Font Awesome Icons for an enhanced user experience 🎨

Toggle Between Login & Sign-Up Forms with JavaScript 🔄

Responsive Layout for better usability 📱
