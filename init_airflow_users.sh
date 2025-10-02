
#!/bin/bash
# init_airflow_users.sh

# 운영 환경 여부: "dev" 또는 "prod" 입력
ENV=${1:-dev}

# 공용 계정 정보 (개발용)
DEV_USERNAME="admin"
DEV_FIRSTNAME="Admin"
DEV_LASTNAME="User"
DEV_ROLE="Admin"
DEV_EMAIL="admin@example.com"
DEV_PASSWORD="admin"

# 운영 계정 정보 (예시: 팀원별)
declare -A PROD_USERS
PROD_USERS["raprok"]="raprok612@gmail.com"
PROD_USERS["johndoe"]="johndoe@example.com"

if [ "$ENV" == "dev" ]; then
    echo "=== 개발용 계정 생성 ==="
    airflow users create \
        --username "$DEV_USERNAME" \
        --firstname "$DEV_FIRSTNAME" \
        --lastname "$DEV_LASTNAME" \
        --role "$DEV_ROLE" \
        --email "$DEV_EMAIL" \
        --password "$DEV_PASSWORD"

elif [ "$ENV" == "prod" ]; then
    echo "=== 운영용 계정 생성 ==="
    for username in "${!PROD_USERS[@]}"; do
        email=${PROD_USERS[$username]}
        airflow users create \
            --username "$username" \
            --firstname "$username" \
            --lastname "User" \
            --role "Admin" \
            --email "$email" \
            --password "ChangeMe123!"
    done
else
    echo "환경 변수를 dev 또는 prod로 설정하세요."
    exit 1
fi
