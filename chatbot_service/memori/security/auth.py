"""
Authentication and Authorization Framework for Memori

This module provides a pluggable authentication/authorization system to prevent
unauthorized access to tenant data.

SECURITY: This is a CRITICAL security component. All Memori instances should
validate that callers are authorized to access the specified user_id, assistant_id,
and session_id.

Usage:
    from memori.security.auth import JWTAuthProvider
    from memori import Memori

    # Create auth provider
    auth = JWTAuthProvider(secret_key=os.getenv("JWT_SECRET"))

    # Create Memori with auth
    memori = Memori(
        database_connect="postgresql://...",
        user_id="user123",
        auth_token=request.headers["Authorization"],
        auth_provider=auth,
    )
"""

from abc import ABC, abstractmethod
from typing import Any
import asyncio
import httpx

from loguru import logger


class AuthenticationError(Exception):
    """Raised when authentication fails"""

    pass


class AuthorizationError(Exception):
    """Raised when authorization check fails"""

    pass


class AuthProvider(ABC):
    """
    Abstract base class for authentication providers.

    Implement this interface to integrate with your authentication system
    (JWT, OAuth, API keys, etc.)
    """

    @abstractmethod
    async def validate_user(self, user_id: str, auth_token: str) -> bool:
        """
        Validate that the auth_token belongs to the specified user_id.

        Args:
            user_id: The user ID being accessed
            auth_token: Authentication token (JWT, API key, etc.)

        Returns:
            True if authentication is valid, False otherwise

        Example:
            # JWT token validation
            payload = jwt.decode(token, secret, algorithms=["HS256"])
            return payload.get("user_id") == user_id
        """
        pass

    @abstractmethod
    async def validate_assistant_access(
        self, user_id: str, assistant_id: str, auth_token: str
    ) -> bool:
        """
        Validate that the user has access to the specified assistant.

        Args:
            user_id: The user ID
            assistant_id: The assistant ID being accessed
            auth_token: Authentication token

        Returns:
            True if user has access to this assistant, False otherwise
        """
        pass

    @abstractmethod
    async def validate_session_access(
        self, user_id: str, session_id: str, auth_token: str
    ) -> bool:
        """
        Validate that the user has access to the specified session.

        Args:
            user_id: The user ID
            session_id: The session ID being accessed
            auth_token: Authentication token

        Returns:
            True if user has access to this session, False otherwise
        """
        pass

    async def extract_user_id(self, auth_token: str) -> str | None:
        """
        Extract user_id from auth token (optional, for convenience).

        Args:
            auth_token: Authentication token

        Returns:
            User ID if extractable, None otherwise
        """
        return None

    async def close(self):
        """Close any open resources (e.g., HTTP clients)."""
        pass


class NoAuthProvider(AuthProvider):
    """
    No-op auth provider that allows all access.

    WARNING: Only use this for development/testing!
    DO NOT use in production!
    """

    def __init__(self):
        logger.warning(
            "WARNING: NoAuthProvider is being used - ALL ACCESS IS ALLOWED! "
            "This should ONLY be used in development. "
            "Use a proper AuthProvider in production!"
        )

    async def validate_user(self, user_id: str, auth_token: str) -> bool:
        return True

    async def validate_assistant_access(
        self, user_id: str, assistant_id: str, auth_token: str
    ) -> bool:
        return True

    async def validate_session_access(
        self, user_id: str, session_id: str, auth_token: str
    ) -> bool:
        return True


class JWTAuthProvider(AuthProvider):
    """
    JWT-based authentication provider.

    Validates JWTs and checks permissions encoded in token claims.

    Example token structure:
        {
            "user_id": "user123",
            "assistants": ["assistant1", "assistant2"],
            "exp": 1234567890
        }
    """

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        """
        Initialize JWT auth provider.

        Args:
            secret_key: Secret key for JWT validation
            algorithm: JWT algorithm (default: HS256)
        """
        self.secret_key = secret_key
        self.algorithm = algorithm

        try:
            import jose.jwt

            self.jwt = jose.jwt
        except ImportError:
            raise ImportError(
                "python-jose is required for JWT auth. "
                "Install with: pip install python-jose[cryptography]"
            )

    def _decode_token(self, auth_token: str) -> dict[str, Any]:
        """Decode and validate JWT token"""
        try:
            # Remove "Bearer " prefix if present
            if auth_token.startswith("Bearer "):
                auth_token = auth_token[7:]

            payload = self.jwt.decode(
                auth_token, self.secret_key, algorithms=[self.algorithm]
            )
            return payload
        except self.jwt.JWTError as e:
            logger.warning(f"JWT validation failed: {e}")
            raise AuthenticationError(f"Invalid authentication token: {e}")
        except Exception as e:
            logger.error(f"Token decoding error: {e}")
            raise AuthenticationError(f"Authentication error: {e}")

    async def validate_user(self, user_id: str, auth_token: str) -> bool:
        """Validate JWT token matches user_id"""
        try:
            payload = self._decode_token(auth_token)
            token_user_id = payload.get("user_id") or payload.get("sub")

            if token_user_id != user_id:
                logger.warning(
                    f"User ID mismatch: token={token_user_id}, requested={user_id}"
                )
                return False

            return True
        except AuthenticationError:
            return False

    async def validate_assistant_access(
        self, user_id: str, assistant_id: str, auth_token: str
    ) -> bool:
        """Validate user has access to assistant"""
        try:
            payload = self._decode_token(auth_token)

            # First validate user
            if not await self.validate_user(user_id, auth_token):
                return False

            # Check assistant permissions
            allowed_assistants = payload.get("assistants", [])

            # If no assistants specified in token, allow all (backward compat)
            if not allowed_assistants:
                return True

            if assistant_id not in allowed_assistants:
                logger.warning(
                    f"User {user_id} not authorized for assistant {assistant_id}"
                )
                return False

            return True
        except AuthenticationError:
            return False

    async def validate_session_access(
        self, user_id: str, session_id: str, auth_token: str
    ) -> bool:
        """Validate user has access to session"""
        try:
            # First validate user
            if not await self.validate_user(user_id, auth_token):
                return False

            # For sessions, we trust that if user is valid, they own their sessions
            # More complex session validation can be added here if needed
            return True
        except AuthenticationError:
            return False

    async def extract_user_id(self, auth_token: str) -> str | None:
        """Extract user_id from JWT"""
        try:
            payload = self._decode_token(auth_token)
            return payload.get("user_id") or payload.get("sub")
        except AuthenticationError:
            return None


class APIKeyAuthProvider(AuthProvider):
    """
    API key-based authentication provider.

    Validates API keys against a database or cache and checks permissions.

    Example:
        auth = APIKeyAuthProvider(
            api_key_validator=lambda key: validate_key_in_database(key)
        )
    """

    def __init__(self, api_key_validator):
        """
        Initialize API key auth provider.

        Args:
            api_key_validator: Callable that takes an API key and returns
                              user info dict or None if invalid
                              Example: {"user_id": "user123", "assistants": [...]}
        """
        self.api_key_validator = api_key_validator

    def _get_user_info(self, auth_token: str) -> dict[str, Any] | None:
        """Get user info from API key"""
        try:
            # Remove "Bearer " or "ApiKey " prefix if present
            if auth_token.startswith(("Bearer ", "ApiKey ")):
                auth_token = auth_token.split(" ", 1)[1]

            user_info = self.api_key_validator(auth_token)
            return user_info
        except Exception as e:
            logger.error(f"API key validation error: {e}")
            return None

    async def validate_user(self, user_id: str, auth_token: str) -> bool:
        """Validate API key matches user_id"""
        user_info = self._get_user_info(auth_token)
        if not user_info:
            return False

        return user_info.get("user_id") == user_id

    async def validate_assistant_access(
        self, user_id: str, assistant_id: str, auth_token: str
    ) -> bool:
        """Validate user has access to assistant"""
        user_info = self._get_user_info(auth_token)
        if not user_info or user_info.get("user_id") != user_id:
            return False

        allowed_assistants = user_info.get("assistants", [])
        if not allowed_assistants:
            return True  # Allow all if none specified

        return assistant_id in allowed_assistants

    async def validate_session_access(
        self, user_id: str, session_id: str, auth_token: str
    ) -> bool:
        """Validate user has access to session"""
        user_info = self._get_user_info(auth_token)
        if not user_info:
            return False

        return user_info.get("user_id") == user_id

    async def extract_user_id(self, auth_token: str) -> str | None:
        """Extract user_id from API key"""
        user_info = self._get_user_info(auth_token)
        return user_info.get("user_id") if user_info else None


class OAuth2AuthProvider(AuthProvider):
    """
    OAuth2-based authentication provider.
    
    Validates OAuth2 access tokens by introspecting with the authorization server
    or validating JWT access tokens directly.
    
    Supports:
    - Token introspection endpoint (RFC 7662)
    - JWT access tokens (self-contained)
    - OpenID Connect userinfo endpoint
    
    Environment Variables:
        OAUTH2_ISSUER: OAuth2/OIDC issuer URL
        OAUTH2_CLIENT_ID: Client ID for token introspection
        OAUTH2_CLIENT_SECRET: Client secret for token introspection
        OAUTH2_JWKS_URI: JWKS endpoint for JWT validation
        OAUTH2_USERINFO_URI: OpenID Connect userinfo endpoint
    
    Example:
        auth = OAuth2AuthProvider(
            issuer="https://auth.example.com",
            client_id="my-client-id",
            client_secret="my-secret"
        )
    """
    
    def __init__(
        self,
        issuer: str = None,
        client_id: str = None,
        client_secret: str = None,
        jwks_uri: str = None,
        userinfo_uri: str = None,
        introspection_uri: str = None,
        audience: str = None,
        use_jwt_validation: bool = True
    ):
        """
        Initialize OAuth2 auth provider.
        
        Args:
            issuer: OAuth2/OIDC issuer URL (used for discovery)
            client_id: Client ID for introspection
            client_secret: Client secret for introspection
            jwks_uri: JWKS endpoint (auto-discovered if issuer provided)
            userinfo_uri: Userinfo endpoint (auto-discovered if issuer provided)
            introspection_uri: Token introspection endpoint
            audience: Expected audience claim
            use_jwt_validation: If True, validate JWT locally; if False, use introspection
        """
        import os
        
        self.issuer = issuer or os.getenv('OAUTH2_ISSUER')
        self.client_id = client_id or os.getenv('OAUTH2_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('OAUTH2_CLIENT_SECRET')
        self.jwks_uri = jwks_uri or os.getenv('OAUTH2_JWKS_URI')
        self.userinfo_uri = userinfo_uri or os.getenv('OAUTH2_USERINFO_URI')
        self.introspection_uri = introspection_uri or os.getenv('OAUTH2_INTROSPECTION_URI')
        self.audience = audience or os.getenv('OAUTH2_AUDIENCE')
        self.use_jwt_validation = use_jwt_validation
        
        # Cache for JWKS keys
        self._jwks_cache: dict[str, Any] | None = None
        self._jwks_cache_time: float = 0
        self._jwks_cache_ttl: float = 3600  # 1 hour
        
        # Token cache for introspection results
        self._token_cache: dict[str, dict[str, Any]] = {}
        
        # Import required libraries
        self.client = httpx.AsyncClient(timeout=10.0)
        
        try:
            from jose import jwt as jose_jwt
            from jose import jwk as jose_jwk
            self._jwt = jose_jwt
            self._jwk = jose_jwk
        except ImportError:
            self._jwt = None
            self._jwk = None
            if use_jwt_validation:
                logger.warning(
                    "python-jose not available for JWT validation. "
                    "Install with: pip install python-jose[cryptography]"
                )
        
        # Endpoints will be discovered lazily or explicitly
        logger.info(f"OAuth2AuthProvider initialized (issuer={self.issuer})")

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def _discover_endpoints(self) -> None:
        """Discover OAuth2/OIDC endpoints from issuer."""
        try:
            # Try OIDC discovery
            discovery_url = f"{self.issuer.rstrip('/')}/.well-known/openid-configuration"
            response = await self.client.get(discovery_url)
            
            if response.status_code == 200:
                config = response.json()
                self.jwks_uri = self.jwks_uri or config.get('jwks_uri')
                self.userinfo_uri = self.userinfo_uri or config.get('userinfo_endpoint')
                self.introspection_uri = self.introspection_uri or config.get('introspection_endpoint')
                logger.info(f"OAuth2 endpoints discovered from {discovery_url}")
            else:
                logger.warning(f"Failed to discover OAuth2 endpoints: {response.status_code}")
        except Exception as e:
            logger.warning(f"OAuth2 endpoint discovery failed: {e}")
    
    async def _get_jwks(self) -> dict[str, Any] | None:
        """Get JWKS (JSON Web Key Set) with caching."""
        import time
        
        # Check cache
        if self._jwks_cache and (time.time() - self._jwks_cache_time) < self._jwks_cache_ttl:
            return self._jwks_cache
        
        if not self.jwks_uri:
            # Try to discover endpoints if not set
            if self.issuer:
                await self._discover_endpoints()
            
            if not self.jwks_uri:
                return None
        
        try:
            response = await self.client.get(self.jwks_uri)
            if response.status_code == 200:
                self._jwks_cache = response.json()
                self._jwks_cache_time = time.time()
                return self._jwks_cache
        except Exception as e:
            logger.error(f"Failed to fetch JWKS: {e}")
        
        return None
    
    async def _validate_jwt_token(self, auth_token: str) -> dict[str, Any] | None:
        """Validate JWT access token using JWKS."""
        if not self._jwt or not self._jwk:
            return None
        
        try:
            # Remove Bearer prefix
            if auth_token.startswith("Bearer "):
                auth_token = auth_token[7:]
            
            # Get JWKS
            jwks = await self._get_jwks()
            if not jwks:
                logger.warning("No JWKS available for JWT validation")
                return None
            
            # Get token header to find key ID
            unverified_header = self._jwt.get_unverified_header(auth_token)
            kid = unverified_header.get('kid')
            
            # Find matching key
            rsa_key = None
            for key in jwks.get('keys', []):
                if key.get('kid') == kid:
                    rsa_key = key
                    break
            
            if not rsa_key:
                logger.warning(f"No matching key found for kid={kid}")
                return None
            
            # Validate token
            options = {
                'verify_aud': bool(self.audience),
                'verify_iss': bool(self.issuer)
            }
            
            payload = self._jwt.decode(
                auth_token,
                rsa_key,
                algorithms=['RS256', 'RS384', 'RS512', 'ES256', 'ES384', 'ES512'],
                audience=self.audience,
                issuer=self.issuer,
                options=options
            )
            
            return payload
            
        except Exception as e:
            logger.warning(f"JWT validation failed: {e}")
            return None
    
    async def _introspect_token(self, auth_token: str) -> dict[str, Any] | None:
        """Introspect token with authorization server (RFC 7662)."""
        if not self.introspection_uri:
            # Try to discover endpoints if not set
            if self.issuer:
                await self._discover_endpoints()
                
            if not self.introspection_uri:
                return None
        
        # Remove Bearer prefix
        if auth_token.startswith("Bearer "):
            auth_token = auth_token[7:]
        
        # Check cache
        if auth_token in self._token_cache:
            cached = self._token_cache[auth_token]
            import time
            if cached.get('_cached_at', 0) + 300 > time.time():  # 5 min cache
                return cached
        
        try:
            auth = (self.client_id, self.client_secret) if self.client_id else None
            response = await self.client.post(
                self.introspection_uri,
                data={'token': auth_token},
                auth=auth
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('active'):
                    import time
                    result['_cached_at'] = time.time()
                    self._token_cache[auth_token] = result
                    return result
            
        except Exception as e:
            logger.error(f"Token introspection failed: {e}")
        
        return None
    
    async def _get_token_claims(self, auth_token: str) -> dict[str, Any] | None:
        """Get token claims via JWT validation or introspection."""
        # Try JWT validation first if enabled
        if self.use_jwt_validation:
            claims = await self._validate_jwt_token(auth_token)
            if claims:
                return claims
        
        # Fall back to introspection
        return await self._introspect_token(auth_token)
    
    async def validate_user(self, user_id: str, auth_token: str) -> bool:
        """Validate OAuth2 token matches user_id."""
        claims = await self._get_token_claims(auth_token)
        if not claims:
            return False
        
        # Check user_id against common claim names
        token_user_id = (
            claims.get('user_id') or 
            claims.get('sub') or 
            claims.get('username') or
            claims.get('preferred_username')
        )
        
        if token_user_id != user_id:
            logger.warning(f"User ID mismatch: token={token_user_id}, requested={user_id}")
            return False
        
        return True
    
    async def validate_assistant_access(
        self, user_id: str, assistant_id: str, auth_token: str
    ) -> bool:
        """Validate user has access to assistant."""
        if not await self.validate_user(user_id, auth_token):
            return False
        
        claims = await self._get_token_claims(auth_token)
        if not claims:
            return False
        
        # Check for assistant permissions in token
        allowed_assistants = claims.get('assistants', [])
        if not allowed_assistants:
            return True  # Allow all if none specified
        
        return assistant_id in allowed_assistants
    
    async def validate_session_access(
        self, user_id: str, session_id: str, auth_token: str
    ) -> bool:
        """Validate user has access to session."""
        return await self.validate_user(user_id, auth_token)
    
    async def extract_user_id(self, auth_token: str) -> str | None:
        """Extract user_id from OAuth2 token."""
        claims = await self._get_token_claims(auth_token)
        if not claims:
            return None
        
        return (
            claims.get('user_id') or 
            claims.get('sub') or 
            claims.get('username') or
            claims.get('preferred_username')
        )


def create_auth_provider(provider_type: str = "jwt", **kwargs) -> AuthProvider:
    """
    Factory function to create auth providers.

    Args:
        provider_type: Type of auth provider ("jwt", "api_key", "oauth2", "none")
        **kwargs: Provider-specific configuration

    Returns:
        AuthProvider instance

    Example:
        # JWT provider
        auth = create_auth_provider("jwt", secret_key="secret")

        # API key provider
        auth = create_auth_provider(
            "api_key",
            api_key_validator=my_validator
        )
        
        # OAuth2 provider
        auth = create_auth_provider(
            "oauth2",
            issuer="https://auth.example.com",
            client_id="my-client-id",
            client_secret="my-secret"
        )

        # Development only - no auth
        auth = create_auth_provider("none")
    """
    providers = {
        "jwt": JWTAuthProvider,
        "api_key": APIKeyAuthProvider,
        "oauth2": OAuth2AuthProvider,
        "none": NoAuthProvider,
    }

    provider_class = providers.get(provider_type)
    if not provider_class:
        raise ValueError(
            f"Unknown auth provider type: {provider_type}. "
            f"Available: {list(providers.keys())}"
        )

    return provider_class(**kwargs)
