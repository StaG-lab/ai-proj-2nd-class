from utils.db import SessionLocal
from models.user import User
import bcrypt


# 비밀번호 해시 처리
def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


# 회원가입 함수
def signup_user(userid, userpw, name, phone_number):
    session = SessionLocal()
    try:
        existing_user = session.query(User).filter(User.userid == userid).first()
        if existing_user:
            return False, "이미 존재하는 사용자 아이디 입니다."

        hashed_pw = hash_password(userpw)
        new_user = User(
            userid=userid,
            userpw=hashed_pw,
            name=name,
            phone_number=phone_number
        )
        session.add(new_user)
        session.commit()
        return True, "회원가입 성공!"
    except Exception as e:
        session.rollback()
        return False, f"회원가입 실패: {e}"
    finally:
        session.close()


# userid 확인
def get_user_by_userid(userid: str):
    session = SessionLocal()
    user = session.query(User).filter(User.userid == userid).first()
    session.close()
    return user


def verify_password(plain_password: str, hashed_pw: str) -> bool:
    return bcrypt.checkpw(plain_password.encode(), hashed_pw.encode())


def authenticate_user(userid: str, userpw: str):
    user = get_user_by_userid(userid)
    if not user:
        return None
    if not verify_password(userpw, user.userpw):
        return None
    return user  # 로그인 성공 시 user 객체 반환