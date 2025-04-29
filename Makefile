run:
	docker-compose up --build

migrate:
	docker-compose run --rm web alembic upgrade head

format:
	ruff check --fix .

lint:
	ruff check .

test:
	docker-compose run --rm web pytest