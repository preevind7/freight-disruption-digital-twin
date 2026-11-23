"""Manage login accounts and the token secret. Run from the backend folder:

    python -m app.manage_users list
    python -m app.manage_users add priya --role dispatcher --name "Priya Deshmukh"
    python -m app.manage_users passwd priya
    python -m app.manage_users remove priya
    python -m app.manage_users seed-demo          # the 4 demo accounts (one per target user)
    python -m app.manage_users secret             # write a new JWT_SECRET into .env

Roles: dispatcher, operator, planner, researcher.
"""
from __future__ import annotations

import argparse
import getpass
import re
import secrets
import sys

from app import config
from app.services import user_store
from app.utils.security import hash_password

DEMO_USERS = [
    ("dispatcher", "Priya Deshmukh", "dispatcher", "Dispatch@123"),
    ("operator", "Arjun Rao", "operator", "Fleet@123"),
    ("planner", "Meera Iyer", "planner", "Planner@123"),
    ("researcher", "Kabir Shah", "researcher", "Research@123"),
]
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")


def _ask_password() -> str:
    while True:
        first = getpass.getpass("New password (min 8 characters): ")
        if len(first) < 8:
            print("Too short, try again.")
            continue
        if first != getpass.getpass("Repeat password: "):
            print("Passwords do not match, try again.")
            continue
        return first


def cmd_list(_):
    users = user_store.list_users()
    if not users:
        print(f"No users in {config.USERS_FILE}. Create some with: python -m app.manage_users seed-demo")
        return
    print(f"{'USERNAME':<16}{'ROLE':<14}{'NAME':<24}STATUS")
    for u in users:
        print(f"{u['username']:<16}{u['role']:<14}{u['name']:<24}{'disabled' if u.get('disabled') else 'active'}")


def cmd_add(args):
    if not USERNAME_RE.match(args.username):
        sys.exit("Username must be 3-32 characters: letters, digits, . _ -")
    if user_store.get_user(args.username):
        sys.exit(f"User '{args.username}' already exists. Use 'passwd' to change the password.")
    password = args.password or _ask_password()
    if len(password) < 8:
        sys.exit("Password must be at least 8 characters.")
    user_store.save_user({"username": args.username.lower(), "name": args.name, "role": args.role, "password_hash": hash_password(password)})
    print(f"Created {args.username} ({args.role}).")


def cmd_passwd(args):
    user = user_store.get_user(args.username)
    if not user:
        sys.exit(f"No user '{args.username}'.")
    password = args.password or _ask_password()
    if len(password) < 8:
        sys.exit("Password must be at least 8 characters.")
    user["password_hash"] = hash_password(password)
    user_store.save_user(user)
    print(f"Password changed for {user['username']}.")


def cmd_remove(args):
    print("Removed." if user_store.delete_user(args.username) else f"No user '{args.username}'.")


def cmd_seed_demo(_):
    for username, name, role, password in DEMO_USERS:
        if user_store.get_user(username):
            print(f"  {username:<12} already exists - skipped")
            continue
        user_store.save_user({"username": username, "name": name, "role": role, "password_hash": hash_password(password)})
        print(f"  {username:<12} password: {password:<14} role: {role}")
    print("Demo accounts ready. Change the passwords before sharing the app: python -m app.manage_users passwd <username>")


def cmd_secret(_):
    env_path = config.BASE_DIR / ".env"
    value = secrets.token_urlsafe(48)
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    lines = [line for line in lines if not line.startswith("JWT_SECRET=")] + [f"JWT_SECRET={value}"]
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"New JWT_SECRET written to {env_path}. Restart uvicorn; existing logins become invalid.")


def main():
    parser = argparse.ArgumentParser(description="Manage login accounts.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list").set_defaults(func=cmd_list)
    add = sub.add_parser("add")
    add.add_argument("username")
    add.add_argument("--role", required=True, choices=list(user_store.ROLES))
    add.add_argument("--name", required=True, help="Full name shown in the app")
    add.add_argument("--password", help="Omit to be prompted (safer: not saved in shell history)")
    add.set_defaults(func=cmd_add)
    pw = sub.add_parser("passwd")
    pw.add_argument("username")
    pw.add_argument("--password")
    pw.set_defaults(func=cmd_passwd)
    rm = sub.add_parser("remove")
    rm.add_argument("username")
    rm.set_defaults(func=cmd_remove)
    sub.add_parser("seed-demo").set_defaults(func=cmd_seed_demo)
    sub.add_parser("secret").set_defaults(func=cmd_secret)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
