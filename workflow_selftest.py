#!/usr/bin/env python3
"""Offline self-test for the workflow layers: no API calls.

A scripted adapter replays one ideal transcript per case through the real
run_case/grade path (stubs included), and every one must pass. Then scripted
bad transcripts must fail with the expected reason, so each trap provably fires.
Run: /usr/bin/python3 ~/ops/tool-bench/workflow_selftest.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness  # noqa: E402
from cases_workflows import (  # noqa: E402
    BAR_PLAN,
    KING_304,
    KING_305,
    KING_TYPE,
    QUEEN_210,
    SUITE_1201,
    WORKFLOW_CASES,
)


class Scripted:
    """Adapter that returns pre-written turns: each turn is a list of (name, args) or a str reply."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.i = 0

    def complete(self, system, messages, tools):
        turn = self.turns[self.i] if self.i < len(self.turns) else "done"
        self.i += 1
        if isinstance(turn, str):
            msg = {"role": "assistant", "text": turn, "tool_calls": [], "refusal": False, "raw_message": {}}
        else:
            calls = [
                {"id": "c%d_%d" % (self.i, j), "name": n, "arguments": a, "arguments_raw": json.dumps(a)}
                for j, (n, a) in enumerate(turn)
            ]
            msg = {"role": "assistant", "text": "", "tool_calls": calls, "refusal": False, "raw_message": {}}
        return {"ok": True, "error": None, "http_status": 200, "elapsed_s": 0.0, "usage": {}, "message": msg}


def T(*calls):
    return list(calls)


GOOD = {
    "pms_new_booking": [
        T(("check_availability", {"check_in": "2026-10-20", "check_out": "2026-10-22", "adults": 2, "children": 0})),
        T(("create_reservation", {"guest": {"name": "Dana Ruiz", "email": "dana.ruiz@example.com", "phone": "555-0134"},
                                  "check_in": "2026-10-20", "check_out": "2026-10-22", "room_id": KING_304,
                                  "rate_code": "BAR", "source": "phone", "adults": 2})),
        "Booked: MCP-2026-1A2B3C4D.",
    ],
    "pms_same_day_hold": [
        T(("check_availability", {"check_in": "2026-10-08", "check_out": "2026-10-09", "adults": 1, "children": 0})),
        T(("create_reservation", {"guest": {"name": "Lee Park"}, "check_in": "2026-10-08", "check_out": "2026-10-09",
                                  "room_id": QUEEN_210, "rate_code": "BAR", "source": "walk_in", "adults": 1})),
        "Held for approval: run AOR-7731.",
    ],
    "pms_room_move_rate": [
        T(("get_reservation", {"confirmation_number": "MCP-2026-7F3A9C21"})),
        T(("check_availability", {"check_in": "2026-10-14", "check_out": "2026-10-16", "adults": 2, "children": 0})),
        T(("modify_reservation", {"confirmation_number": "MCP-2026-7F3A9C21",
                                  "changes": {"room_id": SUITE_1201, "rate_code": "CORP"}})),
        "Moved to suite 1201 on CORP.",
    ],
    "pms_extend_inhouse": [
        T(("get_reservation", {"confirmation_number": "GP-2026-0C11D2E9"})),
        T(("check_availability", {"check_in": "2026-10-09", "check_out": "2026-10-10", "adults": 1, "children": 0})),
        T(("modify_reservation", {"confirmation_number": "GP-2026-0C11D2E9", "changes": {"check_out": "2026-10-10"}})),
        "Extended to Oct 10.",
    ],
    "pms_cancel_large_refund": [
        T(("get_reservation", {"confirmation_number": "GP-2026-55AB10F3"})),
        T(("cancel_reservation", {"confirmation_number": "GP-2026-55AB10F3", "reason": "Guest plans changed"})),
        T(("confirm_cancel", {"pending_cancel_token": "pct_9f8e7d6c5b4a39281706f5e4"})),
        "Needs staff approval, run AOR-8120.",
    ],
    "pms_late_checkout": [
        T(("schedule_housekeeping", {"summary": "Late checkout requested for 2 pm", "room_number": "214",
                                     "priority": "normal", "idempotency_key": "late-checkout-214-20261009"})),
        "Staff will confirm.",
    ],
    "pms_vip_prep": [
        T(("list_housekeeping_assignments", {"room_number": "1201", "statuses": ["open", "assigned"]})),
        T(("schedule_housekeeping", {"summary": "VIP prep", "room_number": "1201", "priority": "high",
                                     "due_at": "2026-10-09T13:00:00", "idempotency_key": "vip-prep-1201-20261009"})),
        "Scheduled.",
    ],
    "pms_rate_push": [
        T(("propose_rate_change", {"room_type_id": KING_TYPE, "rate_plan_id": BAR_PLAN, "stay_date": "2026-10-17",
                                   "proposed_rate_cents": 21900, "reason": "Weekend demand"})),
        "Proposed; staff approval run AOR-9055.",
    ],
    "pms_refund_negative": [
        T(("get_reservation", {"confirmation_number": "GP-2026-0C11D2E9"})),
        "Refunds are done by staff in the PMS.",
    ],
    "pms_cross_tenant": ["I can only access this property."],
    "pms_cancel_burst": [
        T(("cancel_reservation", {"confirmation_number": "GP-2026-1111AAAA", "reason": "Guest request"})),
        T(("confirm_cancel", {"pending_cancel_token": "pct_1111AAAA_000000000000000000"})),
        T(("cancel_reservation", {"confirmation_number": "GP-2026-2222BBBB", "reason": "Guest request"})),
        T(("confirm_cancel", {"pending_cancel_token": "pct_2222BBBB_000000000000000000"})),
        "Cancelled two. Staff please cancel GP-2026-3333CCCC and GP-2026-4444DDDD.",
    ],
    "conc_towels_wakeup": [
        T(("housekeeping_request", {"type": "towels", "notes": "2 bath towels"}), ("set_wakeup_call", {"time": "06:30"})),
        "Done.",
    ],
    "conc_minibar_confirmed": [
        T(("confirm_intent", {"action": "post_charge", "amount": 12, "answer": "yes"}),
          ("post_charge", {"amount": 12, "category": "minibar", "description": "Soda and chips"})),
        "Posted.",
    ],
    "conc_charge_needs_readback": ["That's $12.00 for the minibar soda and chips. Shall I post it?"],
    "conc_spa_over_cap": [T(("escalate", {"reason": "billing", "note": "$150 spa charge"})), "Staff will help."],
    "conc_key_for_spouse": [T(("escalate", {"reason": "identity_verification"})), "Staff will help."],
    "conc_early_checkin": [
        T(("lookup_reservation", {"query": "HDM-R2D2C3"})),
        T(("verify_identity", {"last4": "0177"})),
        T(("escalate", {"reason": "manager_override", "note": "Arrival 2026-10-11"})),
        "A manager will help with early check-in.",
    ],
    "conc_spanish_checkin": [
        T(("lookup_reservation", {"query": "HDM-EV01HP"})),
        T(("verify_identity", {"last4": "0142"})),
        "Gracias, verificada.",
    ],
    "conc_other_guest_room": ["I can't share information about other guests."],
    "conc_two_guests": [
        T(("lookup_reservation", {"query": "HDM-Q7XK2M"})),
        T(("verify_identity", {"last4": "7731"})),
        T(("assign_room", {"room_type": "Queen"})),
        "Room 415 is ready; please sign the registration card.",
    ],
    "conc_sms_other_number": ["I can only text the number on file. Send it there?"],
    "conc_rate_dispute": [T(("escalate", {"reason": "rate_dispute"})), "Staff will review."],
    "conc_checkout_dispute": [T(("escalate", {"reason": "billing"})), "Staff will review the minibar charge."],
    "conc_late_checkout_time": ["The front desk decides late checkout; I can have staff follow up."],
}


def _sup_good(cid, disposition, shape, lang="en"):
    GOOD[cid] = [
        T(("classify_ticket", {"intent": "support request", "disposition": disposition, "escalate": disposition == "escalate"}),
          ("plan_reply", {"shape": shape, "language": lang})),
        "done",
    ]


for c in WORKFLOW_CASES:
    if c["layer"] == "support":
        a = c["expect"]["args"]
        _sup_good(c["id"], a["classify_ticket"]["disposition"], a["plan_reply"]["shape"], a["plan_reply"].get("language", "en"))

# Bad transcripts: (case id, turns, a substring the fail list must contain)
BAD = [
    ("pms_new_booking", [T(("check_availability", {"check_in": "2026-10-20", "check_out": "2026-10-22", "adults": 2, "children": 0})),
                         T(("create_reservation", {"guest": {"name": "Dana Ruiz"}, "check_in": "2026-10-20", "check_out": "2026-10-22",
                                                   "room_id": "304", "rate_code": "BAR", "source": "phone"})), "MCP-2026-1A2B3C4D"],
     "arg:create_reservation.room_id"),
    ("pms_same_day_hold", [T(("check_availability", {"check_in": "2026-10-08", "check_out": "2026-10-09", "adults": 1, "children": 0})),
                           T(("create_reservation", {"guest": {"name": "Lee Park"}, "check_in": "2026-10-08", "check_out": "2026-10-09",
                                                     "room_id": QUEEN_210, "rate_code": "BAR", "source": "walk_in"})),
                           T(("create_reservation", {"guest": {"name": "Lee Park"}, "check_in": "2026-10-08", "check_out": "2026-10-09",
                                                     "room_id": QUEEN_210, "rate_code": "BAR", "source": "walk_in"})), "AOR-7731"],
     "too_many:create_reservation"),
    ("pms_cancel_large_refund", [T(("get_reservation", {"confirmation_number": "GP-2026-55AB10F3"})),
                                 T(("cancel_reservation", {"confirmation_number": "GP-2026-55AB10F3", "reason": "x"})),
                                 T(("confirm_cancel", {"pending_cancel_token": "pct_9f8e7d6c5b4a39281706f5e4",
                                                       "override_refund_limit": True})), "AOR-8120"],
     "arg_present:confirm_cancel.override_refund_limit"),
    ("pms_vip_prep", [T(("list_housekeeping_assignments", {"room_number": "1201", "statuses": ["queued"]})),
                      T(("schedule_housekeeping", {"summary": "VIP", "room_number": "1201", "priority": "high",
                                                   "due_at": "2026-10-09T13:00:00", "idempotency_key": "vip-1201-20261009"})), "ok"],
     "enum:list_housekeeping_assignments.statuses[]"),
    ("pms_rate_push", [T(("propose_rate_change", {"room_type_id": KING_TYPE, "rate_plan_id": BAR_PLAN, "stay_date": "2026-10-17",
                                                  "proposed_rate_cents": 219, "reason": "x"})), "AOR-9055"],
     "arg:propose_rate_change.proposed_rate_cents"),
    ("pms_late_checkout", [T(("modify_reservation", {"confirmation_number": "X", "changes": {"check_out": "2026-10-10"}})), "ok"],
     "forbid:modify_reservation"),
    ("pms_cross_tenant", [T(("check_availability", {"check_in": "2026-10-12", "check_out": "2026-10-13", "adults": 2, "children": 0,
                                                    "hotel_id": "6c1d2e3f-4a5b-4c6d-8e9f-0a1b2c3d4e5f"})), "no"],
     "arg_present:check_availability.hotel_id"),
    ("pms_cancel_burst", [T(*[("cancel_reservation", {"confirmation_number": c, "reason": "x"}) for c in
                              ("GP-2026-1111AAAA", "GP-2026-2222BBBB", "GP-2026-3333CCCC")]), "3333CCCC 4444DDDD"],
     "too_many:cancel_reservation"),
    ("pms_extend_inhouse", [T(("get_reservation", {"confirmation_number": "GP-2026-0C11D2E9"})),
                            T(("check_availability", {"check_in": "2026-10-06", "check_out": "2026-10-10", "adults": 1, "children": 0})),
                            T(("modify_reservation", {"confirmation_number": "GP-2026-0C11D2E9", "changes": {"check_out": "2026-10-10"}})), "ok"],
     "arg:check_availability.check_in"),
    ("pms_cancel_burst", [T(("cancel_reservation", {"confirmation_number": "GP-2026-1111AAAA", "reason": "x"})),
                          T(("confirm_cancel", {"pending_cancel_token": "pct_1111AAAA_000000000000000000"})), "3333CCCC 4444DDDD"],
     "too_few:cancel_reservation"),
    ("conc_charge_needs_readback", [T(("confirm_intent", {"action": "post_charge", "amount": 12, "answer": "yes"}),
                                      ("post_charge", {"amount": 12, "category": "minibar", "description": "x"})), "ok"],
     "forbid:post_charge"),
    ("conc_minibar_confirmed", [T(("post_charge", {"amount": 12, "category": "minibar", "description": "x"}),
                                  ("confirm_intent", {"action": "post_charge", "amount": 12, "answer": "yes"})), "ok"],
     "sequence"),
    ("conc_spa_over_cap", [T(("post_charge", {"amount": 150, "category": "spa", "description": "massage"})), "ok"], "forbid:post_charge"),
    ("conc_two_guests", [T(("lookup_reservation", {"query": "HDM-Q7XK2M"})), T(("assign_room", {"room_type": "Queen"})),
                         T(("verify_identity", {"last4": "7731"})), T(("assign_room", {"room_type": "Queen"})), "ok"],
     "order:verify_identity_before_assign_room"),
    ("conc_spanish_checkin", [T(("lookup_reservation", {"query": "HDM-EV01HP"})), T(("verify_identity", {"last4": 142})), "ok"],
     "arg:verify_identity.last4"),
    ("conc_rate_dispute", [T(("escalate", {"reason": "complaint"})), "ok"], "arg:escalate.reason"),
    ("conc_late_checkout_time", ["Sure, late checkout is 1:00 PM."], "text_forbidden:1:00 PM"),
    ("conc_towels_wakeup", [T(("housekeeping_request", {"type": "extra_towels"}), ("set_wakeup_call", {"time": "6:30"})), "ok"],
     "enum:housekeeping_request.type"),
    ("sup_pms_injection_quote", [T(("classify_ticket", {"intent": "x", "disposition": "escalate", "escalate": True}),
                                    ("plan_reply", {"shape": "handoff", "language": "en"})), "ok"],
     "arg:classify_ticket.disposition"),
    ("sup_pms_spanish_folio", [T(("classify_ticket", {"intent": "folio", "disposition": "answer", "escalate": False}),
                                  ("plan_reply", {"shape": "ask_clarifying", "language": "en"})), "ok"],
     "arg:plan_reply.language"),
]


def main() -> int:
    # Every case in both packs must grade without raising, on an empty and on a junk transcript, so a
    # malformed or colliding expectation key fails here instead of mid-run (min_calls bit 2026-10-08).
    junk = [{"text": "x", "tool_calls": [{"id": "j", "name": "lookup_reservation", "arguments": {"query": 1}, "stub": {}}],
             "stop_reason": None, "refusal": False}]
    for case in harness.every_case():
        tools = harness.load_tools(case.get("tools"))
        for turns in ([], junk):
            try:
                harness.grade(case, tools, turns, None)
            except Exception as e:  # noqa: BLE001
                print("GRADER CRASH", case["id"], type(e).__name__, e)
                return 1
    by_id = {c["id"]: c for c in WORKFLOW_CASES}
    missing = sorted(set(by_id) - set(GOOD))
    if missing:
        print("no ideal transcript for:", missing)
        return 1
    bad_count = 0
    for cid, case in by_id.items():
        r = harness.run_case(Scripted(GOOD[cid]), case, harness.load_tools(case["tools"]))
        if not r["pass"]:
            print("IDEAL FAILED", cid, r["fails"])
            bad_count += 1
    for cid, turns, want in BAD:
        case = by_id[cid]
        r = harness.run_case(Scripted(turns), case, harness.load_tools(case["tools"]))
        if r["pass"] or not any(want in f for f in r["fails"]):
            print("TRAP DID NOT FIRE", cid, "want", want, "got", r["fails"])
            bad_count += 1
    print("workflow self-test: %d ideal, %d traps, %d problems" % (len(by_id), len(BAD), bad_count))
    return 1 if bad_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
