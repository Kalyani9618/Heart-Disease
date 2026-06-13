from fastapi import APIRouter, HTTPException, status, Depends, Header
from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional, Dict, Any
from datetime import timedelta, datetime, timezone
import logging
import re

from core.security import (
    SecurityManager,
    get_current_user,
    hash_password,
    verify_password,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    decode_token
)
from routes.core.auth_db_service import get_auth_db_service, AuthDatabaseService

router = APIRouter(tags=["Authentication"])
logger = logging.getLogger(__name__)

security_manager = SecurityManager()


# Database service (replaces MOCK_USERS_DB)
# ✅ SECURITY FIX: Persistent storage, multi-worker safe
auth_db_service = None


async def get_auth_db() -> 'AuthDatabaseService':
    """Dependency injection for auth database service."""
    global auth_db_service
    if auth_db_service is None:
        auth_db_service = await get_auth_db_service()
    return auth_db_service

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str

    @field_validator('password')
    @classmethod
    def validate_password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    refresh_token: Optional[str] = None
    token: Optional[str] = None
    user: Optional[Dict[str, Any]] = None

class UserResponse(BaseModel):
    id: str  # Accepts int, converts to str
    email: EmailStr
    name: str
    
    @field_validator('id', mode='before')
    @classmethod
    def convert_id_to_string(cls, v):
        """Convert integer id to string for response serialization."""
        return str(v) if v is not None else v

@router.post("/register", response_model=Token)
async def register(user: UserCreate, db: 'AuthDatabaseService' = Depends(get_auth_db)):
    """
    Register a new user.
    
    **Security:**
    - Password hashed with argon2 (OWASP standard)
    - Email uniqueness enforced at database level
    - Multi-worker safe (uses PostgreSQL)
    
    **Returns:**
    - Access token (15 min expiry)
    - Optional refresh token (stored in database/Redis)
    """
    
    # Check if email already registered
    existing_user = await db.get_user_by_email(user.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Hash password with argon2
    hashed_password = hash_password(user.password)
    
    try:
        # Create user in database
        result = await db.register_user(
            email=user.email,
            name=user.name,
            hashed_password=hashed_password
        )
        user_id = result["user_id"]
        
        # Generate access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = security_manager.create_access_token(
            data={"sub": user.email, "user_id": user_id},
            expires_delta=access_token_expires
        )
        
        logger.info(f"✅ User registered (ID: {user_id})")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "token": access_token,
            "user": {
                "id": str(user_id),
                "email": user.email,
                "name": user.name,
                "role": "patient",
            },
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@router.post("/login", response_model=Token)
async def login(user_credentials: UserLogin, db: 'AuthDatabaseService' = Depends(get_auth_db)):
    """
    Authenticate user and issue access token.
    
    **Security:**
    - Rate limiting: Max 3 failed attempts, then 15-min lockout
    - Password verification with argon2
    - Multi-worker safe (queries PostgreSQL)
    
    **Returns:**
    - Access token (15 min expiry)
    """
    
    # Check account lockout
    is_locked = await db.is_account_locked(user_credentials.email)
    if is_locked:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Account locked due to too many failed login attempts. Try again later."
        )
    
    # Fetch user from database
    user = await db.get_user_by_email(user_credentials.email)
    
    if not user or not verify_password(user_credentials.password, user["hashed_password"]):
        # Record failed attempt
        await db.record_failed_attempt(user_credentials.email)
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Successful login - update last login and clear failed attempts
    await db.update_last_login(user_credentials.email)
    
    # Generate access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = security_manager.create_access_token(
        data={"sub": user["email"], "user_id": user["id"]},
        expires_delta=access_token_expires
    )
    
    logger.info(f"✅ User logged in (ID: {user['id']})")
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "token": access_token,
        "user": {
            "id": str(user["id"]),
            "email": user["email"],
            "name": user.get("name", ""),
            "role": "patient",
        },
    }

@router.get("/me", response_model=UserResponse)
async def read_users_me(
    current_user: dict = Depends(get_current_user),
    db: 'AuthDatabaseService' = Depends(get_auth_db)
):
    """
    Get current user profile.
    
    **Security:**
    - Requires valid JWT token
    - Token cannot be revoked (checked at auth middleware)
    """
    
    user_id = current_user.get("user_id")
    
    # Fetch user from database
    user = await db.get_user_by_id(user_id)
    
    if user is None or not user.get("is_active"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {
        "id": user["id"],
        "email": user["email"],
        "name": user["name"]
    }

@router.post("/refresh", response_model=Token)
async def refresh_token(
    authorization: Optional[str] = Header(None),
    db: 'AuthDatabaseService' = Depends(get_auth_db)
):
    """
    Refresh expired access token using refresh token.
    
    **FIX:** This endpoint now:
    1. Accepts expired access tokens (doesn't require get_current_user)
    2. Verifies the token independently (even if expired)
    3. Issues a new access token
    
    **Security:**
    ✅ Works when access_token is expired
    ✅ Validates token signature and user_id
    ✅ Checks token revocation list (logout)
    
    **Usage:**
    ```
    POST /api/auth/refresh
    Authorization: Bearer <expired_access_token>
    ```
    """
    
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header"
        )
    
    token = authorization.split(" ")[1]
    
    # Check if token is revoked (logged out)
    if await db.is_token_revoked(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked (logout)"
        )
    
    try:
        # Decode token (allows expired tokens via verify_exp=False)
        # ✅ FIX: This is the key difference - we accept expired tokens for refresh
        payload = decode_token(token, verify_exp=False)
        
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Enforce maximum refresh window (e.g., 7 days from original issuance)
        issued_at = payload.get("iat")
        if issued_at:
            max_refresh_window = timedelta(days=7)
            token_age = datetime.now(timezone.utc) - datetime.fromtimestamp(issued_at, tz=timezone.utc)
            if token_age > max_refresh_window:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token too old to refresh. Please login again."
                )

        user_email = payload.get("sub")
        user_id = payload.get("user_id")
        
        if not user_email or not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing required claims"
            )
        
        # Verify user still exists
        user = await db.get_user_by_email(user_email)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User no longer exists"
            )
        
        # Generate new access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        new_access_token = security_manager.create_access_token(
            data={"sub": user_email, "user_id": user_id},
            expires_delta=access_token_expires
        )
        
        logger.info("✅ Token refreshed")
        
        return {
            "access_token": new_access_token,
            "token_type": "bearer"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token refresh failed"
        )

@router.post("/logout")
async def logout(
    current_user: dict = Depends(get_current_user),
    authorization: Optional[str] = Header(None),
    db: 'AuthDatabaseService' = Depends(get_auth_db)
):
    """
    Logout user by revoking their access token.
    
    **FIX:** This endpoint now:
    1. Actually revokes the token (adds to blocklist)
    2. Uses Redis for instant invalidation
    3. Token becomes unusable until expiry
    
    **Security:**
    ✅ Logged-out users cannot use their token
    ✅ Token revocation checked on every request
    ✅ Works across all worker processes (Redis is shared)
    
    **Returns:**
    - Success message with revocation details
    """
    
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header"
        )
    
    token = authorization.split(" ")[1]
    
    try:
        # Decode to get expiry
        payload = decode_token(token, verify_exp=False)
        
        # Calculate expiry timestamp from 'exp' claim
        exp_timestamp = datetime.fromtimestamp(payload.get("exp", 0), tz=timezone.utc)
        
        # Add token to revocation list in Redis
        await db.add_to_token_blocklist(token, exp_timestamp)
        
        user_email = current_user.get("sub", "unknown")
        logger.info("✅ User logged out (token revoked)")
        
        return {
            "message": "Successfully logged out",
            "token_revoked": True,
            "revocation_time": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )