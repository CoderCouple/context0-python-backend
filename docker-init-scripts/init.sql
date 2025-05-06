-- > types
CREATE TYPE workflow_status_enum AS ENUM ('DRAFT', 'PUBLISHED');

--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "credential" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" uuid NOT NULL,
	"name" varchar(255) NOT NULL,
	"value" varchar(2048) NOT NULL,
	"created_by" uuid NOT NULL,
	"updated_by" uuid NOT NULL,
	"is_deleted" boolean DEFAULT false NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "execution_log" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"execution_phase_id" uuid NOT NULL,
	"log_level" varchar(20) NOT NULL,
	"message" varchar(2048) NOT NULL,
	"timestamp" timestamp with time zone NOT NULL,
	"created_by" uuid NOT NULL,
	"updated_by" uuid NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL,
	"is_deleted" boolean DEFAULT false NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "execution_phase" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" uuid NOT NULL,
	"workflow_execution_id" uuid NOT NULL,
	"status" varchar(50) NOT NULL,
	"number" integer NOT NULL,
	"node" varchar(255),
	"name" varchar(255),
	"started_at" timestamp with time zone,
	"completed_at" timestamp with time zone,
	"inputs" text,
	"outputs" text,
	"credits_consumed" numeric,
	"created_by" uuid NOT NULL,
	"updated_by" uuid NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL,
	"is_deleted" boolean DEFAULT false NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "user" (
  "id" varchar(64) PRIMARY KEY NOT NULL,
  "name" varchar(256) NOT NULL,
  "email" varchar(320) NOT NULL,
  "password" varchar(256) NOT NULL,
  "role" text DEFAULT 'user' NOT NULL,
  "phone" varchar(256),
  "email_verified" timestamp with time zone,
  "avatar" varchar(2048) NOT NULL,
  CONSTRAINT "user_email_unique" UNIQUE("email")
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "user_purchase" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" uuid NOT NULL,
	"stripe_id" varchar(255) NOT NULL,
	"description" varchar(1024),
	"amount" integer NOT NULL,
	"currency" varchar(10),
	"date" timestamp with time zone,
	"created_by" uuid NOT NULL,
	"updated_by" uuid NOT NULL,
	"is_deleted" boolean DEFAULT false NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "user_purchase_stripe_id_unique" UNIQUE("stripe_id")
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "workflow" (
  "id" varchar(64) PRIMARY KEY NOT NULL,
  "user_id" varchar(64) NOT NULL,
  "name" varchar(255),
  "description" varchar(1024),
  "definition" text,
  "execution_plan" text,
  "cron" varchar(100),
  "status" workflow_status_enum NOT NULL DEFAULT 'DRAFT',
  "credits_cost" numeric,
  "last_run_at" timestamp with time zone,
  "last_run_id" varchar(64),
  "last_run_status" varchar(50),
  "next_run_at" timestamp with time zone,
  "created_by" varchar(64) NOT NULL,
  "updated_by" varchar(64) NOT NULL,
  "is_deleted" boolean DEFAULT false NOT NULL,
  "created_at" timestamp with time zone DEFAULT now() NOT NULL,
  "updated_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "workflow_execution" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"workflow_id" uuid NOT NULL,
	"user_id" uuid NOT NULL,
	"trigger" varchar(255),
	"status" varchar(50),
	"credits_consumed" numeric,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"started_at" timestamp with time zone,
	"completed_at" timestamp with time zone,
	"created_by" uuid NOT NULL,
	"updated_by" uuid NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL,
	"is_deleted" boolean DEFAULT false NOT NULL
);
--> statement-breakpoint
ALTER TABLE "credential" ADD CONSTRAINT "credential_user_id_user_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."user"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "credential" ADD CONSTRAINT "credential_created_by_user_id_fk" FOREIGN KEY ("created_by") REFERENCES "public"."user"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "credential" ADD CONSTRAINT "credential_updated_by_user_id_fk" FOREIGN KEY ("updated_by") REFERENCES "public"."user"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "execution_log" ADD CONSTRAINT "execution_log_execution_phase_id_execution_phase_id_fk" FOREIGN KEY ("execution_phase_id") REFERENCES "public"."execution_phase"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "execution_log" ADD CONSTRAINT "execution_log_created_by_user_id_fk" FOREIGN KEY ("created_by") REFERENCES "public"."user"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "execution_log" ADD CONSTRAINT "execution_log_updated_by_user_id_fk" FOREIGN KEY ("updated_by") REFERENCES "public"."user"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "execution_phase" ADD CONSTRAINT "execution_phase_user_id_user_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."user"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "execution_phase" ADD CONSTRAINT "execution_phase_workflow_execution_id_workflow_execution_id_fk" FOREIGN KEY ("workflow_execution_id") REFERENCES "public"."workflow_execution"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "execution_phase" ADD CONSTRAINT "execution_phase_created_by_user_id_fk" FOREIGN KEY ("created_by") REFERENCES "public"."user"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "execution_phase" ADD CONSTRAINT "execution_phase_updated_by_user_id_fk" FOREIGN KEY ("updated_by") REFERENCES "public"."user"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "user_purchase" ADD CONSTRAINT "user_purchase_user_id_user_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."user"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "user_purchase" ADD CONSTRAINT "user_purchase_created_by_user_id_fk" FOREIGN KEY ("created_by") REFERENCES "public"."user"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "user_purchase" ADD CONSTRAINT "user_purchase_updated_by_user_id_fk" FOREIGN KEY ("updated_by") REFERENCES "public"."user"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "workflow" ADD CONSTRAINT "workflow_user_id_user_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."user"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "workflow" ADD CONSTRAINT "workflow_created_by_user_id_fk" FOREIGN KEY ("created_by") REFERENCES "public"."user"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "workflow" ADD CONSTRAINT "workflow_updated_by_user_id_fk" FOREIGN KEY ("updated_by") REFERENCES "public"."user"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "workflow_execution" ADD CONSTRAINT "workflow_execution_workflow_id_workflow_id_fk" FOREIGN KEY ("workflow_id") REFERENCES "public"."workflow"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "workflow_execution" ADD CONSTRAINT "workflow_execution_user_id_user_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."user"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "workflow_execution" ADD CONSTRAINT "workflow_execution_created_by_user_id_fk" FOREIGN KEY ("created_by") REFERENCES "public"."user"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "workflow_execution" ADD CONSTRAINT "workflow_execution_updated_by_user_id_fk" FOREIGN KEY ("updated_by") REFERENCES "public"."user"("id") ON DELETE no action ON UPDATE no action;


-- Insert into "user"
INSERT INTO "user" (
  id, name, email, password, role, phone, avatar
)
VALUES (
  'f47ac10b-58cc-4372-a567-0e02b2c3d479',  -- valid v4
  'Alice Doe',
  'alice@example.com',
  'hashed_password',
  'admin',
  '+1234567890',
  'https://example.com/avatar.png'
);

-- Insert into "workflow"
INSERT INTO "workflow" (
  id, user_id, name, description, status,
  credits_cost, created_by, updated_by
)
VALUES (
  '1f83caa7-51b4-4d26-aec4-183a0c2cce6c',  -- valid v4
  'f47ac10b-58cc-4372-a567-0e02b2c3d479',
  'Sample Workflow',
  'This is a test workflow.',
  'DRAFT',
  10.00,
  'f47ac10b-58cc-4372-a567-0e02b2c3d479',
  'f47ac10b-58cc-4372-a567-0e02b2c3d479'
);

-- Insert into "workflow_execution"
INSERT INTO "workflow_execution" (
  id, workflow_id, user_id, trigger, status,
  credits_consumed, created_by, updated_by
)
VALUES (
  '85f91f9b-20e8-4cc1-b59e-445fa7c5a708',  -- valid v4
  '1f83caa7-51b4-4d26-aec4-183a0c2cce6c',
  'f47ac10b-58cc-4372-a567-0e02b2c3d479',
  'manual',
  'queued',
  1.0,
  'f47ac10b-58cc-4372-a567-0e02b2c3d479',
  'f47ac10b-58cc-4372-a567-0e02b2c3d479'
);



---- Delete from "workflow_execution"
--DELETE FROM workflow_execution
--WHERE id = '85f91f9b-20e8-4cc1-b59e-445fa7c5a708';
--
---- Delete from "workflow"
--DELETE FROM workflow
--WHERE id = '1f83caa7-51b4-4d26-aec4-183a0c2cce6c';
--
---- Delete from "user"
--DELETE FROM "user"
--WHERE id = 'f47ac10b-58cc-4372-a567-0e02b2c3d479';